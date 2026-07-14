import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models.minimax_m3_eagle1 import _eagle_d2t_offset_to_target_id
from rtp_llm.models_py.model_desc.minimax_m3_eagle1 import MiniMaxM3Eagle1Model
from rtp_llm.models_py.modules.factory.attention.common import (
    target_verify_block_table_for_token_rows,
)


class EagleD2TMappingTest(unittest.TestCase):
    def test_sparse_offset_map_is_converted_to_absolute_ids(self):
        offsets = torch.tensor([0, 0, 0, 100, 0], dtype=torch.int64)

        actual = _eagle_d2t_offset_to_target_id([offsets])

        torch.testing.assert_close(actual, torch.tensor([0, 1, 2, 103, 4]))
        torch.testing.assert_close(offsets, torch.tensor([0, 0, 0, 100, 0]))

    def test_absolute_map_is_preserved(self):
        absolute = torch.tensor([10, 11, 12, 13, 14], dtype=torch.int64)

        actual = _eagle_d2t_offset_to_target_id([absolute])

        torch.testing.assert_close(actual, absolute)

    def test_single_entry_map_is_preserved(self):
        mapping = torch.tensor([7], dtype=torch.int32)

        actual = _eagle_d2t_offset_to_target_id([mapping])

        self.assertEqual(actual.dtype, torch.int64)
        torch.testing.assert_close(actual, torch.tensor([7], dtype=torch.int64))

    def test_rejects_non_vector_map(self):
        with self.assertRaisesRegex(ValueError, "exactly one 1-D tensor"):
            _eagle_d2t_offset_to_target_id([torch.zeros((2, 2), dtype=torch.int64)])

    def test_rejects_multiple_maps(self):
        with self.assertRaisesRegex(ValueError, "exactly one 1-D tensor"):
            _eagle_d2t_offset_to_target_id(
                [torch.zeros(2, dtype=torch.int64), torch.zeros(2, dtype=torch.int64)]
            )


class EagleFcInputTest(unittest.TestCase):
    def _draft(self, hidden_size: int, fc_input_width: int):
        return SimpleNamespace(hidden_size=hidden_size, fc_input_width=fc_input_width)

    def test_hidden_width_input_is_already_packed(self):
        draft = self._draft(hidden_size=4, fc_input_width=4)
        embedding = torch.randn(2, 4)
        hidden = torch.randn(2, 4)
        actual = MiniMaxM3Eagle1Model._build_fc_input(draft, embedding, hidden)
        self.assertIs(actual, hidden)

    def test_double_width_concatenates_embedding_and_hidden(self):
        draft = self._draft(hidden_size=4, fc_input_width=8)
        embedding = torch.randn(2, 4)
        hidden = torch.randn(2, 4)
        actual = MiniMaxM3Eagle1Model._build_fc_input(draft, embedding, hidden)
        torch.testing.assert_close(actual, torch.cat([embedding, hidden], dim=-1))

    def test_triple_width_uses_checkpoint_compatibility_layout(self):
        draft = self._draft(hidden_size=4, fc_input_width=12)
        embedding = torch.randn(2, 4)
        hidden = torch.randn(2, 4)
        actual = MiniMaxM3Eagle1Model._build_fc_input(draft, embedding, hidden)
        torch.testing.assert_close(
            actual, torch.cat([embedding, hidden, hidden], dim=-1)
        )

    def test_rejects_unsupported_fc_width(self):
        draft = self._draft(hidden_size=4, fc_input_width=16)
        with self.assertRaisesRegex(RuntimeError, "Unsupported MiniMax-M3 EAGLE1"):
            MiniMaxM3Eagle1Model._build_fc_input(
                draft, torch.randn(2, 4), torch.randn(2, 4)
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
