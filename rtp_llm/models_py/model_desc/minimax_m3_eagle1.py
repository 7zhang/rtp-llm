from typing import Any, Optional

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules import (
    CausalAttention,
    DenseMLP,
    Embedding,
    RMSNorm,
    RMSResNorm,
)
from rtp_llm.models_py.modules.factory import LinearFactory
from rtp_llm.ops import MoeConfig, ParallelismConfig
from rtp_llm.ops.compute_ops import LayerKVCache, PyModelInputs, PyModelOutputs
from rtp_llm.utils.model_weight import W


class MiniMaxM3Eagle1DecoderLayer(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: dict[str, torch.Tensor],
        layer_idx: int,
        hw_kernel_config: Optional[Any] = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = RMSNorm(
            weights[W.pre_ln_gamma], eps=model_config.layernorm_eps
        )
        self.hidden_norm = RMSNorm(
            weights[W.multi_tokens_predict_hnorm], eps=model_config.layernorm_eps
        )
        self.post_attention_layernorm = RMSResNorm(
            weights[W.post_ln_gamma], eps=model_config.layernorm_eps
        )
        self.self_attn = CausalAttention(
            model_config.attn_config,
            parallelism_config,
            weights,
            model_config.layernorm_eps,
            model_config.quant_config,
            hw_kernel_config,
            layer_idx,
        )
        self.mlp = DenseMLP(
            model_config.activation_type,
            parallelism_config,
            weights,
            model_config.quant_config,
            hw_kernel_config,
        )

    def forward(
        self,
        input_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        fmha_impl: Any,
        kv_cache: Optional[LayerKVCache],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_embeds = self.input_layernorm(input_embeds)
        residual = hidden_states
        hidden_states = self.hidden_norm(hidden_states)
        hidden_states = torch.cat([input_embeds, hidden_states], dim=-1)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            fmha_impl=fmha_impl,
            kv_cache=kv_cache,
        )
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class MiniMaxM3Eagle1Model(GptModelBase):
    def __init__(
        self,
        model_config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: ModelWeights,
        moe_config: MoeConfig,
        max_generate_batch_size: int,
        fmha_config=None,
        py_hw_kernel_config=None,
        device_resource_config=None,
    ):
        super().__init__(
            model_config,
            parallelism_config,
            weights,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=device_resource_config,
        )
        if self.layer_num != 1:
            raise ValueError(
                f"MiniMax-M3 EAGLE1 draft expects one layer, got {self.layer_num}"
            )
        self.embed_tokens = Embedding(
            model_config, parallelism_config, weights.get_global_weight(W.embedding)
        )
        self.fc = LinearFactory.create_linear_from_weights(
            weights.weights[0],
            W.multi_tokens_predict_eh_proj,
            quant_config=model_config.quant_config,
        )
        fc_weight = weights.weights[0][W.multi_tokens_predict_eh_proj]
        self.fc_input_width = int(fc_weight.shape[0])
        self.hidden_size = int(model_config.hidden_size)
        self.layers = nn.ModuleList(
            [
                MiniMaxM3Eagle1DecoderLayer(
                    model_config,
                    parallelism_config,
                    weights.weights[0],
                    0,
                    hw_kernel_config=py_hw_kernel_config,
                )
            ]
        )
        self.norm = RMSResNorm(
            weights.get_global_weight(W.final_ln_gamma), eps=model_config.layernorm_eps
        )

    def clone_for_cuda_graph(self) -> "MiniMaxM3Eagle1Model":
        clone = object.__new__(type(self))
        nn.Module.__init__(clone)
        clone.config = self.config
        clone.parallelism_config = self.parallelism_config
        clone.weight = self.weight
        clone.fmha_config = self.fmha_config
        clone.py_hw_kernel_config = self.py_hw_kernel_config
        clone.micro_batch_size = self.micro_batch_size
        clone.layer_num = self.layer_num
        clone.vocab_size = self.vocab_size
        clone.kv_cache = None
        clone.device_type = self.device_type
        clone.params_dict = {}
        clone.embed_tokens = self.embed_tokens
        clone.fc = self.fc
        clone.fc_input_width = self.fc_input_width
        clone.hidden_size = self.hidden_size
        clone.layers = self.layers
        clone.norm = self.norm
        return clone

    def _build_fc_input(
        self, input_embeds: torch.Tensor, target_hidden: torch.Tensor
    ) -> torch.Tensor:
        hidden_width = int(target_hidden.shape[-1])
        if hidden_width == self.fc_input_width:
            return target_hidden
        if hidden_width != self.hidden_size:
            raise RuntimeError(
                "MiniMax-M3 EAGLE1 draft expected target hidden width "
                f"{self.hidden_size} or prepacked fc width {self.fc_input_width}, "
                f"got {hidden_width}"
            )
        if self.fc_input_width == self.hidden_size * 2:
            return torch.cat([input_embeds, target_hidden], dim=-1)
        if self.fc_input_width == self.hidden_size * 3:
            # Preserve the established checkpoint contract exactly. The three
            # input partitions are token embedding, target final hidden, and the
            # repeated target final hidden. This is a compatibility layout for
            # the temporary checkpoint, not an EAGLE3 auxiliary-hidden layout.
            compatibility_parts = (input_embeds, target_hidden, target_hidden)
            return torch.cat(compatibility_parts, dim=-1)
        raise RuntimeError(
            f"Unsupported MiniMax-M3 EAGLE1 fc input width {self.fc_input_width} "
            f"for hidden size {self.hidden_size}"
        )

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        input_ids: torch.Tensor = inputs.input_ids
        target_hidden = inputs.input_hiddens
        if target_hidden.numel() == 0:
            raise RuntimeError("MiniMax-M3 EAGLE1 draft requires target hidden states")
        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)
        input_embeds = self.embed_tokens(input_ids)
        hidden_states = self.fc(self._build_fc_input(input_embeds, target_hidden))
        residual = torch.zeros_like(hidden_states)
        for i, decoder_layer in enumerate(self.layers[: self.layer_num]):
            hidden_states, residual = decoder_layer(
                input_embeds,
                hidden_states,
                fmha_impl,
                self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return PyModelOutputs(hidden_states, fmha_impl.fmha_params)
