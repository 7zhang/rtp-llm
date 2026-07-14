import functools
import json
import os
from typing import List

import torch

from rtp_llm.config.model_config import ModelConfig as PyModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.model_loader.attn_weight import AttnAtomicWeight, AttnConfig
from rtp_llm.model_loader.ffn_weight import FfnAtomicWeight, FfnConfig, FfnWeight
from rtp_llm.model_loader.model_weight_info import (
    ModelDeployWeightInfo,
    ModelWeightInfo,
)
from rtp_llm.model_loader.weight_module import AtomicWeight
from rtp_llm.models.llama import Llama
from rtp_llm.utils.model_weight import (
    CkptWeightInfo,
    W,
    concat_0,
    concat_1,
    identity,
    transpose,
    zeros,
)


class MiniMaxM3Eagle1WeightNames:
    WQ = "midlayer.self_attn.q_proj.weight"
    WK = "midlayer.self_attn.k_proj.weight"
    WV = "midlayer.self_attn.v_proj.weight"
    WO = "midlayer.self_attn.o_proj.weight"
    FFW1 = "midlayer.mlp.gate_proj.weight"
    FFW2 = "midlayer.mlp.down_proj.weight"
    FFW3 = "midlayer.mlp.up_proj.weight"
    ATTEN_NORM = "midlayer.input_layernorm.weight"
    FFN_NORM = "midlayer.post_attention_layernorm.weight"
    HIDDEN_NORM = "midlayer.hidden_norm.weight"
    TOKEN_EMBEDDING = "embed_tokens.weight"
    NORM = "norm.weight"
    OUTPUT = "lm_head.weight"
    FC = "fc.weight"
    D2T = "d2t"
    T2D = "t2d"


def _merge_qkv_hf(ts: List[torch.Tensor], hidden_size, head_num_kv, head_num):
    q, k, v = ts
    return torch.concat([q.T, k.T, v.T], dim=1).contiguous()


def _eagle_d2t_offset_to_target_id(ts: List[torch.Tensor]) -> torch.Tensor:
    if len(ts) != 1 or ts[0].dim() != 1:
        raise ValueError("MiniMax-M3 EAGLE1 d2t must contain exactly one 1-D tensor")
    mapping = ts[0].to(torch.int64).contiguous()
    if mapping.numel() <= 1:
        return mapping
    # Current MiniMax-M3 EAGLE1 checkpoints store a sparse offset map:
    # identity entries are zero and only remapped draft ids are nonzero.
    # Inspect the whole map so an early remapped id cannot be mistaken for an
    # absolute map. This runs once while loading weights, outside inference.
    zero_count = int((mapping == 0).sum().item())
    looks_like_offset = zero_count * 2 > mapping.numel()
    if not looks_like_offset:
        return mapping
    base = torch.arange(mapping.numel(), dtype=torch.int64, device=mapping.device)
    return (base + mapping).contiguous()


class MiniMaxM3Eagle1WeightInfo(ModelDeployWeightInfo):
    def _process_meta(self, meta_dicts, weight_keys):
        if MiniMaxM3Eagle1WeightNames.FC not in weight_keys:
            raise Exception(
                "unknown MiniMax-M3 EAGLE1 weights format: missing fc.weight"
            )
        self._names = MiniMaxM3Eagle1WeightNames
        self._merge_qkv = _merge_qkv_hf

    def _get_weight_info(self):
        names = self._names
        attn_config = AttnConfig(
            hidden_size=self._hidden_size,
            size_per_head=self._size_per_head,
            head_num=self._head_num,
            head_num_kv=self._head_num_kv,
        )
        ffn_config = FfnConfig(
            is_gated_activation=True,
            align_size=self._align_size,
            is_moe=False,
        )
        weights = [
            AtomicWeight(
                W.embedding,
                [CkptWeightInfo(names.TOKEN_EMBEDDING, concat_1)],
                identity,
            ),
            AtomicWeight(
                W.final_ln_gamma,
                [CkptWeightInfo(names.NORM, identity)],
                identity,
            ),
            AtomicWeight(
                W.final_ln_beta,
                [],
                functools.partial(zeros, shape=[self._hidden_size]),
            ),
            AtomicWeight(
                W.lm_head,
                [CkptWeightInfo(names.OUTPUT, concat_0)],
                identity,
            ),
            AtomicWeight(
                W.multi_tokens_predict_d2t_map,
                [CkptWeightInfo(names.D2T, identity)],
                _eagle_d2t_offset_to_target_id,
                data_type=torch.int64,
            ),
            AtomicWeight(
                W.multi_tokens_predict_t2d_map,
                [CkptWeightInfo(names.T2D, identity)],
                identity,
                data_type=torch.int64,
            ),
        ]
        layer_weights = [
            AtomicWeight(
                W.multi_tokens_predict_eh_proj,
                [CkptWeightInfo(names.FC, identity)],
                transpose,
            ),
            AtomicWeight(
                W.multi_tokens_predict_hnorm,
                [CkptWeightInfo(names.HIDDEN_NORM, identity)],
                identity,
            ),
            AtomicWeight(
                W.multi_tokens_predict_enorm,
                [CkptWeightInfo(names.ATTEN_NORM, identity)],
                identity,
            ),
            AtomicWeight(
                W.pre_ln_gamma,
                [CkptWeightInfo(names.ATTEN_NORM, identity)],
                identity,
            ),
            AtomicWeight(
                W.post_ln_gamma,
                [CkptWeightInfo(names.FFN_NORM, identity)],
                identity,
            ),
            AttnAtomicWeight(
                W.attn_o_w,
                [CkptWeightInfo(names.WO, concat_1)],
                transpose,
                config=attn_config,
            ),
            FfnWeight(
                sub_weights=[
                    FfnAtomicWeight(
                        W.ffn_w1,
                        [CkptWeightInfo(names.FFW1, concat_0)],
                        transpose,
                        config=ffn_config,
                    ),
                    FfnAtomicWeight(
                        W.ffn_w3,
                        [CkptWeightInfo(names.FFW3, concat_0)],
                        transpose,
                        config=ffn_config,
                    ),
                    FfnAtomicWeight(
                        W.ffn_w2,
                        [CkptWeightInfo(names.FFW2, concat_1)],
                        transpose,
                        config=ffn_config,
                    ),
                ],
                config=ffn_config,
            ),
            AttnAtomicWeight(
                W.attn_qkv_w,
                [
                    CkptWeightInfo(names.WQ, concat_0),
                    CkptWeightInfo(names.WK, concat_0),
                    CkptWeightInfo(names.WV, concat_0),
                ],
                functools.partial(
                    self._merge_qkv,
                    hidden_size=self._hidden_size,
                    head_num_kv=self._head_num_kv,
                    head_num=self._head_num,
                ),
                config=attn_config,
            ),
        ]
        return ModelWeightInfo(layer_weights=[layer_weights], weights=weights)


class MiniMaxM3Eagle1(Llama):
    @classmethod
    def _create_config(cls, ckpt_path: str) -> PyModelConfig:
        config = PyModelConfig()
        config.ckpt_path = ckpt_path
        config.attn_config.rope_config.dim = 128
        config.attn_config.rope_config.style = 1
        config_path = os.path.join(ckpt_path, "config.json")
        if not os.path.exists(config_path):
            raise Exception("MiniMax-M3 EAGLE1 parameter from unknown source")
        with open(config_path) as reader:
            config_json = json.loads(reader.read())
        Llama.from_huggingface(config, config_json)
        rope_parameters = config_json.get("rope_parameters") or {}
        if "rope_theta" in rope_parameters:
            config.attn_config.rope_config.base = int(rope_parameters["rope_theta"])
        config.vocab_size = int(config_json.get("draft_vocab_size", config.vocab_size))
        config.activation_type = "SiGLU"
        config.model_type = "minimax_m3_eagle1"
        return config

    @staticmethod
    def get_weight_cls():
        return MiniMaxM3Eagle1WeightInfo

    def support_cuda_graph(self) -> bool:
        return True

    def _create_python_model(self):
        from rtp_llm.models_py.model_desc.minimax_m3_eagle1 import MiniMaxM3Eagle1Model

        self.py_model = MiniMaxM3Eagle1Model(
            self.model_config,
            self.parallelism_config,
            self.weight,
            self.moe_config,
            max_generate_batch_size=self.max_generate_batch_size,
            fmha_config=self.fmha_config,
            py_hw_kernel_config=self.hw_kernel_config,
            device_resource_config=self.device_resource_config,
        )
        return self.py_model


register_model(
    "minimax_m3_eagle1",
    MiniMaxM3Eagle1,
    ["MiniMaxM3Eagle1ForCausalLM", "LlamaForCausalLMEagle3"],
)
