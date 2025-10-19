# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Union

import torch
from einops import rearrange
from fla.modules import ShortConvolution
from fla.modules.l2norm import l2_norm
from fla.ops.path_attn.parallel import parallel_path_attn
from megatron.core import tensor_parallel
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.models.common.embeddings.rope_utils import (
    apply_rotary_pos_emb,
)
from megatron.core.process_groups_config import ModelCommProcessGroups
from megatron.core.transformer.attention import Attention
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from torch import Tensor

from .enums import AttnMaskType
from .transformer_config import TransformerConfig

try:
    import transformer_engine  # pylint: disable=unused-import

    HAVE_TE = True
    from megatron.core.extensions.transformer_engine import SplitAlongDim
except ImportError:
    HAVE_TE = False
    SplitAlongDim = None


@dataclass
class PaTHAttentionSubmodules:
    """
    Configuration class for specifying the submodules of a self-attention.
    """

    linear_qkvwb: Union[ModuleSpec, type] = None
    core_attention: Union[ModuleSpec, type] = None
    linear_proj: Union[ModuleSpec, type] = None
    q_layernorm: Union[ModuleSpec, type] = None
    k_layernorm: Union[ModuleSpec, type] = None


class ParallelPaTHAttention(MegatronModule):

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: float = None,
        softmax_scale: float = None,
        cp_comm_type: str = None,
        model_comm_pgs: ModelCommProcessGroups = None,
    ):
        super().__init__(config=config)

        self.config: TransformerConfig = config

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        weight: Tensor,
        beta: Tensor,
        gate: Optional[Tensor] = None,
        packed_seq_params = None,
    ):
        """Forward."""
        cu_seqlens = packed_seq_params.cu_seqlens_kv if packed_seq_params is not None else None
        return parallel_path_attn(
            q=query,
            k=key,
            v=value,
            w=weight,
            beta=beta,
            g=gate,
            cu_seqlens=cu_seqlens,
        )[0]


class PaTHAttention(Attention):

    def __init__(
        self,
        config: TransformerConfig,
        submodules: PaTHAttentionSubmodules,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding,
        cp_comm_type: str = None,
        model_comm_pgs: ModelCommProcessGroups = None,
        impl: Literal["full", "full_rope"] = "full",
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            attention_type="self",
            cp_comm_type=cp_comm_type,
            model_comm_pgs=model_comm_pgs,
        )

        size_qkvwb = self.query_projection_size + 3 * self.kv_projection_size + self.config.num_query_groups
        self.linear_qkvwb = build_module(
            submodules.linear_qkvwb,
            self.config.hidden_size,
            size_qkvwb,
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

        self.attn_token_shift = config.attn_token_shift
        if self.attn_token_shift == "shortconv":
            self.w_conv1d = ShortConvolution(
                hidden_size=self.kv_projection_size,
                kernel_size=config.shortconv_kernel_size,
                bias=False,
                activation='silu',
                backend='cuda',
            )
        else:
            self.w_conv1d = None

        self.path_rope = (impl == "full_rope")

    def get_query_key_value_tensors(self,
                                    hidden_states,
                                    key_value_states=None):
        """
        Derives `query`, `key`, `value`, `weight` and `beta` tensors from `hidden_states`.
        """
        # Attention heads [b, sq, h] --> [b, sq, ng * (np/ng + 3) * hn + 1)]
        mixed_qkvwb, _ = self.linear_qkvwb(hidden_states.transpose(0, 1))

        # [b, sq, hp] --> [b, sq, ng, (np/ng + 3) * hn + 1]
        new_tensor_shape = mixed_qkvwb.size()[:-1] + (
            self.num_query_groups_per_partition,
            ((self.num_attention_heads_per_partition //
              self.num_query_groups_per_partition + 3) *
             self.hidden_size_per_attention_head + 1),
        )
        mixed_qkvwb = mixed_qkvwb.view(*new_tensor_shape)

        split_arg_list = [
            (self.num_attention_heads_per_partition //
             self.num_query_groups_per_partition *
             self.hidden_size_per_attention_head),
            self.hidden_size_per_attention_head,
            self.hidden_size_per_attention_head,
            self.hidden_size_per_attention_head,
            1,
        ]

        if SplitAlongDim is not None:

            # [b, sq, ng, (np/ng + 3) * hn + 1]
            # --> [b, sq, ng, np/ng * hn], [b, sq, ng, hn], [b, sq, ng, hn], [b, sq, ng, hn], [b, sq, ng, 1]
            (query, key, value, weight,
             beta) = SplitAlongDim(mixed_qkvwb, 3, split_arg_list)
        else:

            # [b, sq, ng, (np/ng + 3) * hn + 1]
            # --> [b, sq, ng, np/ng * hn], [b, sq, ng, hn], [b, sq, ng, hn], [b, b, sq, ng, hn], [b, sq, ng, 1]
            (query, key, value, weight, beta) = torch.split(mixed_qkvwb,
                                                            split_arg_list,
                                                            dim=3)

        # [b, sq, ng, 1] --> [b, sq, ng]
        beta = beta.squeeze(3)

        # allowing negative eigenvalues
        beta = beta.float().sigmoid() * 2

        # [b, sq, ng, np/ng * hn] -> [b, sq, np, hn]
        query = query.reshape(query.size(0), query.size(1), -1,
                              self.hidden_size_per_attention_head)

        if self.q_layernorm is not None:
            query = self.q_layernorm(query)

        if self.k_layernorm is not None:
            key = self.k_layernorm(key)

        if self.config.test_mode:
            self.run_realtime_tests()

        return query, key, value, weight, beta

    def _checkpointed_attention_forward(
        self,
        query,
        key,
        value,
        weight,
        beta,
        packed_seq_params,
    ):
        """Forward method with selective activation checkpointing."""

        def custom_forward(*inputs):
            query = inputs[0]
            key = inputs[1]
            value = inputs[2]
            weight = inputs[3]
            beta = inputs[4]
            packed_seq_params = inputs[5]
            output_ = self.core_attention(
                query,
                key,
                value,
                weight,
                beta,
                packed_seq_params,
            )
            return output_

        hidden_states = tensor_parallel.checkpoint(custom_forward, False,
                                                   query, key, value, weight,
                                                   beta, packed_seq_params)

        return hidden_states

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        key_value_states: Optional[Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        rotary_pos_emb: Optional[Union[Tensor, Tuple[Tensor, Tensor]]] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[int] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        """
        Perform a forward pass through the attention module.

        Args:
            hidden_states (Tensor): Hidden states.
            attention_mask (Tensor): Attention mask.
            key_value_states (Optional[Tensor]): Key/value states (for cross attention).
            inference_context (Optional[BaseInferenceContext]): Inference context that manages
                KV cache.
            rotary_pos_emb (Optional[Union[Tensor, Tuple[Tensor, Tensor]]]): Rotary
                embedding tensor(s).
            rotary_pos_cos (Optional[Tensor]): Rotary embedding cosine.
            rotary_pos_sin (Optional[Tensor]): Rotary embedding sine.
            attention_bias (Optional[Tensor]): Attention bias.
            packed_seq_params (Optional[PackedSeqparams]): Parameters used for THD format.
            sequence_len_offset (Optional[int]): Sequence length offset used for
                inference CUDA graphs.

        Return:
            (Tuple[Tensor, Tensor]) Attention output and bias.

        """

        assert key_value_states is None, "Cross attention is not supported in path attention"
        assert attention_bias is None, "Attention bias is not supported in path attention"
        assert all(o is None for o in [inference_context, rotary_pos_cos, rotary_pos_sin, sequence_len_offset]), "Inference is not supported in path attention"

        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb,) * 2

        # =====================
        # Query, Key, and Value
        # =====================
        # Get the query, key and value tensors based on the type of attention -
        # self or cross attn.
        query, key, value, weight, beta = self.get_query_key_value_tensors(
            hidden_states)
        if self.w_conv1d:
            cu_seqlens = packed_seq_params.cu_seqlens_kv if packed_seq_params is not None else None
            weight, _ = self.w_conv1d(
                rearrange(weight, 'b s n h -> b s (n h)'),
                cache=None,
                output_final_state=False,
                cu_seqlens=cu_seqlens,
            )
            weight = rearrange(weight, 'b s (n h) -> b s n h', n=self.num_attention_heads_per_partition)
        weight = l2_norm(weight, output_dtype=torch.float32)

        # Training
        if attention_mask is not None:
            raise ValueError("attention_mask is not supported in training")

        # if packed_seq_params is not None:
        #     query = query.squeeze(1)
        #     key = key.squeeze(1)
        #     value = value.squeeze(1)

        # ================================================
        # relative positional embedding (rotary embedding)
        # ================================================

        if self.path_rope and rotary_pos_emb is not None and not self.config.flash_decode:
            q_pos_emb, k_pos_emb = rotary_pos_emb

            if packed_seq_params is not None:
                if packed_seq_params.cu_seqlens_q_padded is not None:
                    cu_seqlens_q = packed_seq_params.cu_seqlens_q_padded
                else:
                    cu_seqlens_q = packed_seq_params.cu_seqlens_q
                if packed_seq_params.cu_seqlens_kv_padded is not None:
                    cu_seqlens_kv = packed_seq_params.cu_seqlens_kv_padded
                else:
                    cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
            else:
                cu_seqlens_q = cu_seqlens_kv = None

            if q_pos_emb is not None:
                query = apply_rotary_pos_emb(
                    query,
                    q_pos_emb,
                    config=self.config,
                    cu_seqlens=cu_seqlens_q,
                    cp_group=self.model_comm_pgs.cp,
                )
            if k_pos_emb is not None:
                key = apply_rotary_pos_emb(
                    key,
                    k_pos_emb,
                    config=self.config,
                    cu_seqlens=cu_seqlens_kv,
                    cp_group=self.model_comm_pgs.cp,
                )

            # TODO, can apply positional embedding to value_layer so it has
            # absolute positional embedding.
            # otherwise, only relative positional embedding takes effect
            # VO RoPE: https://kexue.fm/archives/10862
            # value_layer = apply_rotary_pos_emb(value_layer, k_pos_emb)

        # ==================================
        # core attention computation
        # ==================================
        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query,
                key,
                value,
                weight,
                beta,
                packed_seq_params=packed_seq_params,
            )
        else:
            core_attn_out = self.core_attention(
                query,
                key,
                value,
                weight,
                beta,
                packed_seq_params=packed_seq_params,
            )

        if packed_seq_params is not None and packed_seq_params.qkv_format == 'thd':
            # reshape to same output shape as unpacked case
            # (t, np, hn) -> (t, b=1, h=np*hn)
            # t is the pack size = sum (sq_i)
            # note that batch is a dummy dimension in the packed case
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

        # =================
        # Output. [b, sq, ng, hn] -> [sq, b, h]
        # =================
        core_attn_out = rearrange(core_attn_out, 'b s n h -> s b (n h)')
        output, bias = self.linear_proj(core_attn_out)

        return output, bias
