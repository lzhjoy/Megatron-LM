# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from megatron.core.extensions.transformer_engine import (
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TERowParallelLinear,
)
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.gpt.gpt_layer_specs import get_mlp_module_spec
from megatron.core.ssm.mamba_block import MambaStack, MambaStackSubmodules
from megatron.core.ssm.mamba_layer import MambaLayer, MambaLayerSubmodules
from megatron.core.ssm.mamba_mixer import MambaMixer, MambaMixerSubmodules
from megatron.core.ssm.mlp_layer import MLPLayer
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.gated_attention import GatedSelfAttention, GatedSelfAttentionSubmodules
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.torch_norm import L2Norm
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules

try:
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TEDotProductAttention,
        TELayerNormColumnParallelLinear,
        TENorm,
        TERowParallelLinear,
    )

    HAVE_TE = True
except ImportError:
    HAVE_TE = False


mamba_stack_spec = ModuleSpec(
    module=MambaStack,
    submodules=MambaStackSubmodules(
        mamba_layer=ModuleSpec(
            module=MambaLayer,
            submodules=MambaLayerSubmodules(
                mixer=ModuleSpec(
                    module=MambaMixer,
                    submodules=MambaMixerSubmodules(
                        in_proj=TELayerNormColumnParallelLinear, out_proj=TERowParallelLinear
                    ),
                ),
                mamba_bda=get_bias_dropout_add,
            ),
        ),
        # Started with spec from gpt_layer_specs.py (with MLP removed)
        # Using the TE spec because we had problems getting the non-TE spec
        # working
        attention_layer=ModuleSpec(
            module=TransformerLayer,
            submodules=TransformerLayerSubmodules(
                self_attention=ModuleSpec(
                    module=SelfAttention,
                    params={"attn_mask_type": AttnMaskType.causal},
                    submodules=SelfAttentionSubmodules(
                        linear_qkv=TELayerNormColumnParallelLinear,
                        core_attention=TEDotProductAttention,
                        linear_proj=TERowParallelLinear,
                    ),
                ),
                self_attn_bda=get_bias_dropout_add,
            ),
        ),
        # Started with spec from gpt_layer_specs.py
        # Using the TE spec because we had problems getting the non-TE spec
        # working
        mlp_layer=ModuleSpec(
            module=MLPLayer,
            submodules=TransformerLayerSubmodules(
                mlp=ModuleSpec(
                    module=MLP,
                    submodules=MLPSubmodules(
                        linear_fc1=TELayerNormColumnParallelLinear, linear_fc2=TERowParallelLinear
                    ),
                ),
                mlp_bda=get_bias_dropout_add,
            ),
        ),
    ),
)

def mamba_moe_stack_spec(config: TransformerConfig):
    assert HAVE_TE

    if config.attn_output_gate is not None:
        self_attn_module = GatedSelfAttention
        self_attn_submodules = GatedSelfAttentionSubmodules
        self_attn_kwargs = {
            "linear_gate_down_proj": TEColumnParallelLinear if config.attn_output_gate == 'lora' else None,
            "linear_gate_up_proj": TERowParallelLinear if config.attn_output_gate == 'lora' else None,
        }
    else:
        self_attn_module = SelfAttention
        self_attn_submodules = SelfAttentionSubmodules
        self_attn_kwargs = {}

    # If we want to log the output hidden states of input_layernorm,
    # we have to split TELayerNormColumnParallelLinear op.
    if isinstance(config.log_layer_hidden_states, list) and "input_layernorm" in config.log_layer_hidden_states:
        input_layernorm = TENorm
        linear_qkv = TEColumnParallelLinear
    else:
        input_layernorm = IdentityOp
        linear_qkv = TELayerNormColumnParallelLinear

    return ModuleSpec(
        module=MambaStack,
        submodules=MambaStackSubmodules(
            mamba_layer=ModuleSpec(
                module=MambaLayer,
                submodules=MambaLayerSubmodules(
                    mixer=ModuleSpec(
                        module=MambaMixer,
                        submodules=MambaMixerSubmodules(
                            in_proj=TELayerNormColumnParallelLinear, out_proj=TERowParallelLinear
                        ),
                    ),
                    mamba_bda=get_bias_dropout_add,
                ),
            ),
            # Started with spec from gpt_layer_specs.py (with MLP removed)
            # Using the TE spec because we had problems getting the non-TE spec
            # working
            attention_layer=ModuleSpec(
                module=TransformerLayer,
                submodules=TransformerLayerSubmodules(
                    input_layernorm=input_layernorm,
                    self_attention=ModuleSpec(
                        module=self_attn_module,
                        params={"attn_mask_type": AttnMaskType.causal},
                        submodules=self_attn_submodules(
                            linear_qkv=linear_qkv,
                            core_attention=TEDotProductAttention,
                            linear_proj=TERowParallelLinear,
                            q_layernorm=(
                                L2Norm if config.qk_l2_norm else (TENorm if config.qk_layernorm else IdentityOp)
                            ),
                            k_layernorm=(
                                L2Norm if config.qk_l2_norm else (TENorm if config.qk_layernorm else IdentityOp)
                            ),
                            **self_attn_kwargs,
                        ),
                    ),
                    self_attn_bda=get_bias_dropout_add,
                ),
            ),
            # Started with spec from gpt_layer_specs.py
            # Using the TE spec because we had problems getting the non-TE spec
            # working
            mlp_layer=ModuleSpec(
                module=MLPLayer,
                submodules=TransformerLayerSubmodules(
                    pre_mlp_layernorm=TENorm if config.num_moe_experts else IdentityOp,
                    mlp=get_mlp_module_spec(
                        num_experts=config.num_moe_experts,
                        moe_grouped_gemm=config.moe_grouped_gemm,
                    ),
                    mlp_bda=get_bias_dropout_add,
                ),
            ),
        ),
    )
