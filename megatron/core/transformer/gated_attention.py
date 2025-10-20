# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass
from typing import Optional, Union

import torch

from megatron.core.parallel_state import (
    get_data_parallel_group,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.attention import Attention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import ModuleSpec, build_module

from .transformer_config import TransformerConfig

try:
    import transformer_engine  # pylint: disable=unused-import

    HAVE_TE = True
    from megatron.core.extensions.transformer_engine import SplitAlongDim
except ImportError:
    HAVE_TE = False
    SplitAlongDim = None

@dataclass
class GatedSelfAttentionSubmodules:
    """Submodules for the MLA self-attention layer."""

    linear_qkv: Union[ModuleSpec, type] = None
    core_attention: Union[ModuleSpec, type] = None
    linear_proj: Union[ModuleSpec, type] = None
    q_layernorm: Union[ModuleSpec, type] = None
    k_layernorm: Union[ModuleSpec, type] = None
    linear_gate_down_proj: Union[ModuleSpec, type] = None
    linear_gate_up_proj: Union[ModuleSpec, type] = None


class GatedSelfAttention(Attention):
    """Multi-Latent Attention layer abstract class.

    This layer only contains common modules required for the "self attn" and
    "cross attn" specializations.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: GatedSelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding,
        cp_comm_type: str = None,
        pg_collection: ProcessGroupCollection = None,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type="self",
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
        )

        self.lora_gate = config.attn_output_gate == "lora"
        if self.lora_gate:
            self.config.g_lora_rank = self.config.hidden_size // 4
            assert submodules.linear_gate_down_proj is not None and submodules.linear_gate_up_proj is not None
            self.linear_gate_down_proj = build_module(
                submodules.linear_gate_down_proj,
                self.config.hidden_size,
                self.config.g_lora_rank,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=False,
                skip_bias_add=False,
                is_expert=False,
            )
            self.linear_gate_up_proj = build_module(
                submodules.linear_gate_up_proj,
                self.config.g_lora_rank,
                self.query_projection_size,
                config=self.config,
                init_method=self.config.init_method,
                bias=False,
                input_is_parallel=True,
                skip_bias_add=False,
                is_expert=False,
            )

            qkv_projection_size = self.query_projection_size + 2 * self.kv_projection_size
        else:
            qkv_projection_size = 2 * self.query_projection_size + 2 * self.kv_projection_size

        self.linear_qkv = build_module(
            submodules.linear_qkv,
            self.config.hidden_size,
            qkv_projection_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear or self.config.add_qkv_bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='qkv',
        )

        if submodules.q_layernorm is not None:
            self.q_layernorm = build_module(
                submodules.q_layernorm,
                hidden_size=self.hidden_size_per_attention_head,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.q_layernorm = None

        if submodules.k_layernorm is not None:
            self.k_layernorm = build_module(
                submodules.k_layernorm,
                hidden_size=self.hidden_size_per_attention_head,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.k_layernorm = None

    @torch.compile(dynamic=True)
    def apply_attn_output_gate(self, context_layer, output_gate):
        output_gate = torch.sigmoid(output_gate.reshape(context_layer.size(0), context_layer.size(1), -1))
        context_layer = context_layer * output_gate
        return context_layer

    @torch.compile(dynamic=False, disable=True)
    def get_gate_tensor_lora(self, hidden_states):
        compressed, _ = self.linear_gate_down_proj(hidden_states)
        gate, _ = self.linear_gate_up_proj(compressed)
        return gate

    def get_query_key_value_tensors(self, hidden_states, key_value_states=None):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`.
        """
        # Attention heads [sq, b, h] --> [sq, b, ng * (np/ng + 2) * hn)]
        mixed_qkv, _ = self.linear_qkv(hidden_states)

        # [sq, b, hp] --> [sq, b, ng, (np/ng + 2) * hn]
        query_gate_size = 1 + int(not self.lora_gate)
        new_tensor_shape = mixed_qkv.size()[:-1] + (
            self.num_query_groups_per_partition,
            (
                (query_gate_size * self.num_attention_heads_per_partition // self.num_query_groups_per_partition + 2)
                * self.hidden_size_per_attention_head
            ),
        )
        mixed_qkv = mixed_qkv.view(*new_tensor_shape)

        split_arg_list = [
            (
                query_gate_size * self.num_attention_heads_per_partition
                // self.num_query_groups_per_partition
                * self.hidden_size_per_attention_head
            ),
            self.hidden_size_per_attention_head,
            self.hidden_size_per_attention_head,
        ]

        if SplitAlongDim is not None:

            # [sq, b, ng, (np/ng + 2) * hn]
            # --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
            (query, key, value) = SplitAlongDim(mixed_qkv, 3, split_arg_list)
        else:

            # [sq, b, ng, (np/ng + 2) * hn]
            # --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
            (query, key, value) = torch.split(mixed_qkv, split_arg_list, dim=3)

        # get gate
        if self.lora_gate:
            gate = self.get_gate_tensor_lora(hidden_states)
        else:
            query, gate = torch.chunk(query, 2, dim=-1)

        # [sq, b, ng, np/ng * hn] -> [sq, b, np, hn]
        query = query.reshape(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)

        if self.q_layernorm is not None:
            query = self.q_layernorm(query)

        if self.k_layernorm is not None:
            key = self.k_layernorm(key)

        if self.config.test_mode:
            self.run_realtime_tests()

        return query, key, value, gate
