import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch

from rtp_llm.models.minimax_m3_eagle1 import (
    _external_lm_head_path,
    _identity_d2t_map,
    _load_external_lm_head,
)
from rtp_llm.models_py.model_desc.minimax_m3_eagle1 import MiniMaxM3Eagle1Model
from rtp_llm.models_py.modules.factory.attention.common import (
    target_verify_block_table_for_token_rows,
)


class EagleIdentityMappingTest(unittest.TestCase):
    def test_full_vocab_d2t_identity_map_is_generated_for_hass_checkpoint(self):
        torch.testing.assert_close(
            _identity_d2t_map([], vocab_size=5),
            torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64),
        )


class EagleExternalLmHeadTest(unittest.TestCase):
    def test_loads_lm_head_from_bundle_assets_sibling(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ckpt = root / "draft_model"
            assets = root / "assets"
            ckpt.mkdir()
            assets.mkdir()
            expected = torch.randn(3, 4, dtype=torch.bfloat16)
            torch.save(expected, assets / "lm_head.pt")

            self.assertEqual(
                _external_lm_head_path(str(ckpt)), str(assets / "lm_head.pt")
            )
            actual = _load_external_lm_head([], ckpt_path=str(ckpt))
            torch.testing.assert_close(actual, expected)

    def test_rejects_missing_lm_head(self):
        with TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir) / "draft_model"
            ckpt.mkdir()
            with self.assertRaisesRegex(FileNotFoundError, "external lm_head"):
                _load_external_lm_head([], ckpt_path=str(ckpt))


class EagleFcInputTest(unittest.TestCase):
    def test_hass_input_normalizes_embedding_and_hidden_before_projection(self):
        draft = SimpleNamespace(
            hidden_size=4,
            embedding_norm=lambda value: value + 1,
            hidden_norm=lambda value: value * 2,
        )
        embedding = torch.randn(2, 4)
        hidden = torch.randn(2, 4)

        actual = MiniMaxM3Eagle1Model._build_fc_input(draft, embedding, hidden)

        torch.testing.assert_close(
            actual, torch.cat([embedding + 1, hidden * 2], dim=-1)
        )

    def test_hass_input_rejects_wrong_target_hidden_width(self):
        draft = SimpleNamespace(
            hidden_size=4,
            embedding_norm=lambda value: value,
            hidden_norm=lambda value: value,
        )
        with self.assertRaisesRegex(RuntimeError, "HASS draft expected target hidden"):
            MiniMaxM3Eagle1Model._build_fc_input(
                draft, torch.randn(2, 4), torch.randn(2, 5)
            )


class TargetVerifyBlockTableTest(unittest.TestCase):
    def test_expands_request_rows_to_verify_token_rows(self):
        inputs = SimpleNamespace(
            is_target_verify=True,
            prefix_lengths=torch.tensor([10, 20], dtype=torch.int32),
            sequence_lengths_plus_1_d=torch.tensor(
                [11, 12, 13, 21, 22, 23], dtype=torch.int32
            ),
        )
        table = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
        actual = target_verify_block_table_for_token_rows(inputs, table)
        expected = torch.tensor(
            [[1, 2], [1, 2], [1, 2], [3, 4], [3, 4], [3, 4]],
            dtype=torch.int32,
        )
        torch.testing.assert_close(actual, expected)

    def test_rejects_non_divisible_token_rows(self):
        inputs = SimpleNamespace(
            is_target_verify=True,
            prefix_lengths=torch.zeros(2, dtype=torch.int32),
            sequence_lengths_plus_1_d=torch.zeros(5, dtype=torch.int32),
        )
        with self.assertRaisesRegex(RuntimeError, "must be divisible"):
            target_verify_block_table_for_token_rows(
                inputs, torch.zeros((2, 3), dtype=torch.int32)
            )

    def test_rejects_wrong_block_rows_for_single_verify_token(self):
        inputs = SimpleNamespace(
            is_target_verify=True,
            prefix_lengths=torch.zeros(2, dtype=torch.int32),
            sequence_lengths_plus_1_d=torch.ones(2, dtype=torch.int32),
        )
        with self.assertRaisesRegex(RuntimeError, "block table row mismatch"):
            target_verify_block_table_for_token_rows(
                inputs, torch.zeros((1, 3), dtype=torch.int32)
            )


if __name__ == "__main__":
    unittest.main()
