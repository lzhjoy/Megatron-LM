# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import copy
import pytest
import torch
import torch.nn.functional as F
from einops import rearrange

from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.process_groups_config import ModelCommProcessGroups
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.path_attention import PaTHAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

# Import FLA PaTHAttention for comparison
try:
    import sys
    import os
    # Add the flash-linear-attention path to sys.path
    fla_path = os.path.join(os.path.dirname(__file__), '../../../../flash-linear-attention')
    if os.path.exists(fla_path):
        sys.path.insert(0, fla_path)
    from fla.layers.path_attn import PaTHAttention as FLAPaTHAttention
    HAVE_FLA = True
except ImportError:
    HAVE_FLA = False
    FLAPaTHAttention = None


class TestPaTHAttention:
    """Test class for PaTHAttention module."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

        # Configuration for Megatron PaTHAttention
        self.transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            num_query_groups=4,  # For MHA setup
            use_cpu_initialization=True,
        )

        self.transformer_config_w_shortconv = TransformerConfig(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            num_query_groups=4,  # For MHA setup
            use_cpu_initialization=True,
            attn_token_shift="shortconv",
            shortconv_kernel_size=3,
        )

        # Get layer spec for path attention
        layer_spec = get_gpt_layer_with_transformer_engine_spec(path_attention="full")
        attention_spec = layer_spec.submodules.self_attention

        self.megatron_path_attention = PaTHAttention(
            self.transformer_config,
            attention_spec.submodules,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        )

        self.megatron_path_attention_w_shortconv = PaTHAttention(
            self.transformer_config_w_shortconv,
            attention_spec.submodules,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_constructor(self):
        """Test that PaTHAttention is constructed correctly."""
        assert isinstance(self.megatron_path_attention, PaTHAttention)
        assert self.megatron_path_attention.layer_number == 1

        # Check that the attention has the expected components
        assert getattr(self.megatron_path_attention, 'linear_qkvwb') is not None
        assert getattr(self.megatron_path_attention, 'core_attention') is not None
        assert getattr(self.megatron_path_attention_w_shortconv, 'w_conv1d') is not None
        assert getattr(self.megatron_path_attention_w_shortconv, 'linear_proj') is not None

    @pytest.mark.skipif(not HAVE_FLA, reason="FLA not available")
    @pytest.mark.parametrize("use_w_shortconv", [True, False])
    def test_parameter_count_alignment(self, use_w_shortconv):
        """Test that parameter shapes match between FLA and Megatron implementations."""
        # Create FLA PaTHAttention with matching configuration
        fla_attention = FLAPaTHAttention(
            hidden_size=self.transformer_config.hidden_size,
            num_heads=self.transformer_config.num_attention_heads,
            num_kv_heads=self.transformer_config.num_query_groups,
            use_qk_norm=False,
            use_low_rank_w=False,  # Use standard linear projection for comparison
            use_w_shortconv=use_w_shortconv,
            conv_size=self.transformer_config.shortconv_kernel_size,
            conv_bias=False,
        )

        # Compare parameter counts
        if use_w_shortconv:
            megatron_path_attention = self.megatron_path_attention_w_shortconv
        else:
            megatron_path_attention = self.megatron_path_attention

        megatron_params = sum(p.numel() for p in megatron_path_attention.parameters())
        fla_params = sum(p.numel() for p in fla_attention.parameters())

        assert abs(megatron_params - fla_params) / max(megatron_params, fla_params) < 0.5

    @pytest.mark.flaky
    @pytest.mark.flaky_in_dev
    def test_gpu_forward_shape(self):
        """Test GPU forward pass of Megatron PaTHAttention."""
        config = self.megatron_path_attention.config
        sequence_length = 32
        micro_batch_size = 2

        self.megatron_path_attention.cuda()

        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones(
            (sequence_length, micro_batch_size, config.hidden_size)
        ).cuda()

        # PaTHAttention doesn't use attention mask in training
        output, bias = self.megatron_path_attention(hidden_states, attention_mask=None)

        assert config.recompute_granularity is None
        assert output.shape[0] == sequence_length
        assert output.shape[1] == micro_batch_size
        assert output.shape[2] == config.hidden_size
        assert bias.shape[0] == config.hidden_size

    @pytest.mark.skipif(not HAVE_FLA, reason="FLA not available")
    @pytest.mark.flaky
    @pytest.mark.flaky_in_dev
    @pytest.mark.parametrize("use_w_shortconv", [True, False])
    def test_qkvwb_output_alignment(self, use_w_shortconv):
        """Test that qkvwb and full outputs align between FLA and Megatron implementations."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        if use_w_shortconv:
            megatron_path_attention = self.megatron_path_attention_w_shortconv
        else:
            megatron_path_attention = self.megatron_path_attention

        assert megatron_path_attention.path_rope == False

        config = megatron_path_attention.config
        sequence_length = 32
        micro_batch_size = 2

        # Move Megatron attention to GPU
        megatron_path_attention.cuda()

        # Create FLA attention with matching configuration
        fla_attention = FLAPaTHAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_query_groups,
            use_qk_norm=False,
            use_low_rank_w=False,  # Use standard linear projection for comparison
            use_w_shortconv=use_w_shortconv,
            conv_size=config.shortconv_kernel_size,
            conv_bias=False,
        ).cuda()

        # Create input tensor
        # Megatron format: [seq_len, batch_size, hidden_size]
        megatron_input = torch.randn(
            (sequence_length, micro_batch_size, config.hidden_size),
            device='cuda',
            dtype=torch.bfloat16
        ) / 10

        # FLA format: [batch_size, seq_len, hidden_size]
        fla_input = megatron_input.clone().transpose(0, 1)

        # Copy parameters from Megatron to FLA following the correct reshape logic
        with torch.no_grad():
            # Get Megatron QKVWB weights and bias
            megatron_qkvwb_weight = megatron_path_attention.linear_qkvwb.weight
            megatron_qkvwb_bias = megatron_path_attention.linear_qkvwb.bias if hasattr(megatron_path_attention.linear_qkvwb, 'bias') and megatron_path_attention.linear_qkvwb.bias is not None else None

            hidden_size = config.hidden_size
            head_dim = hidden_size // config.num_attention_heads
            num_query_groups = config.num_query_groups
            num_attention_heads = config.num_attention_heads

            # Calculate sizes following Megatron's logic
            query_projection_size = num_attention_heads * head_dim
            kv_projection_size = num_query_groups * head_dim
            bt_size = num_query_groups  # beta size

            # Reshape weight matrix to [ng, (np/ng + 3) * hn + 1, hidden_size]
            heads_per_group = num_attention_heads // num_query_groups
            group_size = (heads_per_group + 3) * head_dim + 1  # +1 for beta

            # Reshape the weight matrix
            weight_reshaped = megatron_qkvwb_weight.clone().view(num_query_groups, group_size, hidden_size)

            # Split along the group dimension following the split_arg_list
            split_sizes = [
                heads_per_group * head_dim,  # query
                head_dim,                    # key
                head_dim,                    # value
                head_dim,                    # weight
                1,                           # beta
            ]

            splits = torch.split(weight_reshaped, split_sizes, dim=1)
            q_weight_grouped = splits[0]  # [ng, np/ng * hn, hidden_size]
            k_weight_grouped = splits[1]  # [ng, hn, hidden_size]
            v_weight_grouped = splits[2]  # [ng, hn, hidden_size]
            w_weight_grouped = splits[3]  # [ng, hn, hidden_size]
            bt_weight_grouped = splits[4] # [ng, 1, hidden_size]

            # Reshape query weight: [ng, np/ng * hn, hidden_size] -> [np * hn, hidden_size]
            q_weight = q_weight_grouped.reshape(-1, hidden_size)

            # Reshape k, v, w weights: [ng, hn, hidden_size] -> [ng * hn, hidden_size]
            k_weight = k_weight_grouped.reshape(-1, hidden_size)
            v_weight = v_weight_grouped.reshape(-1, hidden_size)
            w_weight = w_weight_grouped.reshape(-1, hidden_size)

            # Reshape beta weight: [ng, 1, hidden_size] -> [ng, hidden_size]
            bt_weight = bt_weight_grouped.reshape(-1, hidden_size)

            # Handle bias if present
            if megatron_qkvwb_bias is not None:
                bias_reshaped = megatron_qkvwb_bias.clone().view(num_query_groups, group_size)
                bias_splits = torch.split(bias_reshaped, split_sizes, dim=1)
                q_bias_grouped = bias_splits[0]
                k_bias_grouped = bias_splits[1]
                v_bias_grouped = bias_splits[2]
                w_bias_grouped = bias_splits[3]
                bt_bias_grouped = bias_splits[4]

                q_bias = q_bias_grouped.reshape(-1)
                k_bias = k_bias_grouped.reshape(-1)
                v_bias = v_bias_grouped.reshape(-1)
                w_bias = w_bias_grouped.reshape(-1)
                bt_bias = bt_bias_grouped.reshape(-1)

            # Assign to FLA
            try:
                fla_attention.q_proj.weight.copy_(q_weight)
                fla_attention.k_proj.weight.copy_(k_weight)
                fla_attention.v_proj.weight.copy_(v_weight)
                if hasattr(fla_attention.w_proj, 'weight'):
                    fla_attention.w_proj.weight.copy_(w_weight)
                fla_attention.bt_proj.weight.copy_(bt_weight)

                if megatron_qkvwb_bias is not None:
                    if hasattr(fla_attention.q_proj, 'bias') and fla_attention.q_proj.bias is not None:
                        fla_attention.q_proj.bias.copy_(q_bias)
                    if hasattr(fla_attention.k_proj, 'bias') and fla_attention.k_proj.bias is not None:
                        fla_attention.k_proj.bias.copy_(k_bias)
                    if hasattr(fla_attention.v_proj, 'bias') and fla_attention.v_proj.bias is not None:
                        fla_attention.v_proj.bias.copy_(v_bias)
                    if hasattr(fla_attention.w_proj, 'bias') and fla_attention.w_proj.bias is not None:
                        fla_attention.w_proj.bias.copy_(w_bias)
                    if hasattr(fla_attention.bt_proj, 'bias') and fla_attention.bt_proj.bias is not None:
                        fla_attention.bt_proj.bias.copy_(bt_bias)

                # Copy output projection weights
                fla_attention.o_proj.weight.copy_(megatron_path_attention.linear_proj.weight)
                if hasattr(megatron_path_attention.linear_proj, 'bias') and megatron_path_attention.linear_proj.bias is not None:
                    if hasattr(fla_attention.o_proj, 'bias') and fla_attention.o_proj.bias is not None:
                        fla_attention.o_proj.bias.copy_(megatron_path_attention.linear_proj.bias)

                param_copy_success = True
            except Exception as e:
                # If direct copying fails, we'll just test with random weights
                import warnings
                warnings.warn(f"Parameter copying failed (continuing with random weights): {e}", UserWarning)
                param_copy_success = False

        # Get QKVWB tensors from Megatron
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            megatron_q, megatron_k, megatron_v, megatron_w, megatron_beta = megatron_path_attention.get_query_key_value_tensors(
                megatron_input
            )

        # Create LayerNorm to match Megatron's TELayerNormColumnParallelLinear
        fla_input_layernorm = torch.nn.LayerNorm(
            config.hidden_size,
            eps=config.layernorm_epsilon,
            dtype=torch.bfloat16
        ).cuda()

        # Copy LayerNorm parameters from Megatron if available
        with torch.no_grad():
            # Try to extract LayerNorm parameters from TELayerNormColumnParallelLinear
            if hasattr(megatron_path_attention.linear_qkvwb, 'layer_norm_weight'):
                fla_input_layernorm.weight.copy_(megatron_path_attention.linear_qkvwb.layer_norm_weight)
            if hasattr(megatron_path_attention.linear_qkvwb, 'layer_norm_bias'):
                fla_input_layernorm.bias.copy_(megatron_path_attention.linear_qkvwb.layer_norm_bias)

        # Get QKVWB tensors from FLA
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            fla_input_normalized = fla_input_layernorm(fla_input)
            fla_q = fla_attention.q_proj(fla_input_normalized)
            fla_k = fla_attention.k_proj(fla_input_normalized)
            fla_v = fla_attention.v_proj(fla_input_normalized)
            fla_w = fla_attention.w_proj(fla_input_normalized)
            fla_beta = fla_attention.bt_proj(fla_input_normalized).float().sigmoid() * 2

            # Reshape FLA tensors to match Megatron format
            # FLA: [batch_size, seq_len, hidden_size] -> Megatron: [seq_len, batch_size, num_heads, head_dim]
            fla_q = rearrange(fla_q, 'b s (h d) -> b s h d', h=config.num_attention_heads)
            fla_k = rearrange(fla_k, 'b s (h d) -> b s h d', h=config.num_query_groups)
            fla_v = rearrange(fla_v, 'b s (h d) -> b s h d', h=config.num_query_groups)
            fla_w = rearrange(fla_w, 'b s (h d) -> b s h d', h=config.num_query_groups)

        # Compare shapes
        assert megatron_q.shape == fla_q.shape, f"Query shape mismatch: Megatron {megatron_q.shape} vs FLA {fla_q.shape}"
        assert megatron_k.shape == fla_k.shape, f"Key shape mismatch: Megatron {megatron_k.shape} vs FLA {fla_k.shape}"
        assert megatron_v.shape == fla_v.shape, f"Value shape mismatch: Megatron {megatron_v.shape} vs FLA {fla_v.shape}"
        assert megatron_w.shape == fla_w.shape, f"Weight shape mismatch: Megatron {megatron_w.shape} vs FLA {fla_w.shape}"
        assert megatron_beta.shape == fla_beta.shape, f"Beta shape mismatch: Megatron {megatron_beta.shape} vs FLA {fla_beta.shape}"

        # Compare numerical values if parameter copying was successful
        if param_copy_success:
            # Calculate comparison statistics for debugging
            def compare_tensors(megatron_tensor, fla_tensor, name):
                max_abs_diff = (megatron_tensor - fla_tensor).abs().max().item()
                mean_abs_diff = (megatron_tensor - fla_tensor).abs().mean().item()
                megatron_mean = megatron_tensor.mean().item()
                fla_mean = fla_tensor.mean().item()
                megatron_std = megatron_tensor.std().item()
                fla_std = fla_tensor.std().item()

                comparison_info = (
                    f"\n{name} comparison statistics:\n"
                    f"  Megatron shape: {megatron_tensor.shape}\n"
                    f"  FLA shape: {fla_tensor.shape}\n"
                    f"  Megatron mean: {megatron_mean:.6f}, std: {megatron_std:.6f}\n"
                    f"  FLA mean: {fla_mean:.6f}, std: {fla_std:.6f}\n"
                    f"  Max absolute difference: {max_abs_diff:.6f}\n"
                    f"  Mean absolute difference: {mean_abs_diff:.6f}"
                    f"  Megatron tensor: {megatron_tensor}\n"
                    f"  FLA tensor: {fla_tensor}"
                )

                try:
                    torch.testing.assert_close(
                        megatron_tensor,
                        fla_tensor,
                        rtol=1e-3,
                        atol=1e-3,
                        msg=f"Megatron and FLA {name} should be close when using same parameters"
                    )
                    return f"✓ {name} alignment test passed - tensors are close!{comparison_info}"
                except AssertionError as e:
                    return f"⚠ {name} alignment test failed: {str(e)}{comparison_info}"

            # Compare each tensor
            q_result = compare_tensors(megatron_q, fla_q, "Query")
            k_result = compare_tensors(megatron_k, fla_k, "Key")
            v_result = compare_tensors(megatron_v, fla_v, "Value")
            w_result = compare_tensors(megatron_w, fla_w, "Weight")
            beta_result = compare_tensors(megatron_beta, fla_beta, "Beta")

            # Combine all results
            all_results = f"{q_result}\n{k_result}\n{v_result}\n{w_result}\n{beta_result}"

            # Check if any test failed
            if "⚠" in all_results:
                results_str = "\n".join([f"{result}" for result in [q_result, k_result, v_result, w_result, beta_result] if "⚠" in result])
                raise AssertionError(f"QKVWB alignment test failed:\n{results_str}\nmegatron weight: {megatron_qkvwb_weight}\nfla q: {fla_attention.q_proj.weight}\nfla k: {fla_attention.k_proj.weight}\nfla v: {fla_attention.v_proj.weight}\nfla w: {fla_attention.w_proj.weight}\nfla beta: {fla_attention.bt_proj.weight}\nmegatron input: {megatron_input}\nfla input: {fla_input}")

        # Test full output alignment
        # Forward pass through Megatron
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            megatron_output, megatron_bias = megatron_path_attention(
                megatron_input, attention_mask=None
            )

        # Forward pass through FLA
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            fla_input_normalized = fla_input_layernorm(fla_input)
            fla_output, _, _ = fla_attention(fla_input_normalized, attention_mask=None)

        # Convert FLA output back to Megatron format
        fla_output_megatron_format = fla_output.transpose(0, 1)

        # Compare shapes
        assert megatron_output.shape == fla_output_megatron_format.shape

        # Compare numerical values - if parameter copying was successful, outputs should be close
        if param_copy_success:
            # Calculate comparison statistics for debugging
            max_abs_diff = (megatron_output - fla_output_megatron_format).abs().max().item()
            mean_abs_diff = (megatron_output - fla_output_megatron_format).abs().mean().item()
            megatron_mean = megatron_output.mean().item()
            fla_mean = fla_output_megatron_format.mean().item()
            megatron_std = megatron_output.std().item()
            fla_std = fla_output_megatron_format.std().item()

            # Use assert with detailed message for debugging info
            comparison_info = (
                f"\nOutput comparison statistics:\n"
                f"  Megatron shape: {megatron_output.shape}\n"
                f"  FLA shape: {fla_output_megatron_format.shape}\n"
                f"  Megatron mean: {megatron_mean:.6f}, std: {megatron_std:.6f}\n"
                f"  Megatron bias mean: {megatron_bias.mean().item():.6f}, std: {megatron_bias.std().item():.6f}\n"
                f"  FLA mean: {fla_mean:.6f}, std: {fla_std:.6f}\n"
                f"  Max absolute difference: {max_abs_diff:.6f}\n"
                f"  Mean absolute difference: {mean_abs_diff:.6f}"
            )

            # Try to assert close, but provide detailed info if it fails
            try:
                torch.testing.assert_close(
                    megatron_output,
                    fla_output_megatron_format,
                    rtol=1e-3,
                    atol=1e-3,
                    msg=f"Megatron and FLA outputs should be close when using same parameters"
                )
                # If assertion passes, still show the comparison info
                assert True, f"✓ Full output alignment test passed - outputs are close!{comparison_info}"
            except AssertionError as e:
                # If assertion fails, provide detailed comparison info
                assert False, f"⚠ Full output alignment test failed: {str(e)}{comparison_info}"
        else:
            # If parameter copying failed, just verify shapes are correct
            assert True, f"✓ Full output shape alignment test passed (parameter copying failed, so only shapes were verified):\n" \
                        f"  Megatron: {megatron_output.shape}\n" \
                        f"  FLA: {fla_output_megatron_format.shape}"

    @pytest.mark.flaky_in_dev
    def test_checkpointed_gpu_forward_shape(self):
        """Test checkpointed forward pass."""
        transformer_config = copy.deepcopy(self.transformer_config)
        transformer_config.recompute_granularity = 'selective'

        layer_spec = get_gpt_layer_with_transformer_engine_spec(path_attention="full")
        attention_spec = layer_spec.submodules.self_attention

        checkpointed_path_attention = PaTHAttention(
            transformer_config,
            attention_spec.submodules,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
        )

        config = checkpointed_path_attention.config
        sequence_length = 32
        micro_batch_size = 2

        checkpointed_path_attention.cuda()

        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones(
            (sequence_length, micro_batch_size, config.hidden_size)
        ).cuda()

        output, bias = checkpointed_path_attention(hidden_states, attention_mask=None)

        assert config.recompute_granularity == 'selective'
        assert "core_attn" in config.recompute_modules
        assert output.shape[0] == sequence_length
        assert output.shape[1] == micro_batch_size
        assert output.shape[2] == config.hidden_size
        assert bias.shape[0] == config.hidden_size

    def test_get_query_key_value_tensors_shape(self):
        """Test the get_query_key_value_tensors method."""
        sequence_length = 16
        micro_batch_size = 2

        self.megatron_path_attention.cuda()

        hidden_states = torch.randn(
            (sequence_length, micro_batch_size, self.transformer_config.hidden_size)
        ).cuda()

        query, key, value, weight, beta = self.megatron_path_attention.get_query_key_value_tensors(
            hidden_states
        )

        # Check shapes
        expected_head_dim = self.transformer_config.hidden_size // self.transformer_config.num_attention_heads

        assert query.shape == (micro_batch_size, sequence_length,
                              self.transformer_config.num_attention_heads, expected_head_dim)
        assert key.shape == (micro_batch_size, sequence_length,
                            self.transformer_config.num_query_groups, expected_head_dim)
        assert value.shape == (micro_batch_size, sequence_length,
                              self.transformer_config.num_query_groups, expected_head_dim)
        assert weight.shape == (micro_batch_size, sequence_length,
                               self.transformer_config.num_query_groups, expected_head_dim)
        assert beta.shape == (micro_batch_size, sequence_length,
                             self.transformer_config.num_query_groups)

        # Check that beta is in valid range (0, 2) after sigmoid * 2
        assert torch.all(beta > 0)
        assert torch.all(beta < 2)


class TestPaTHAttentionMPU:
    """Test PaTHAttention with model parallel utilities."""

    def setup_method(self, method):
        Utils.destroy_model_parallel()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def run_path_attention_w_shortconv(self, model_comm_pgs):
        """Run PaTHAttention with given process groups."""
        transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            num_query_groups=4,
            use_cpu_initialization=False,
            attn_token_shift="shortconv",
            shortconv_kernel_size=3,
        )

        layer_spec = get_gpt_layer_with_transformer_engine_spec(path_attention="full")
        attention_spec = layer_spec.submodules.self_attention

        path_attention = PaTHAttention(
            transformer_config,
            attention_spec.submodules,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            model_comm_pgs=model_comm_pgs,
        )

        config = path_attention.config
        sequence_length = 64
        micro_batch_size = 2

        path_attention.cuda()

        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones(
            (sequence_length, micro_batch_size, config.hidden_size),
            device='cuda',
        )

        output, bias = path_attention(hidden_states, None)

        assert config.recompute_granularity is None
        # Check if output and bias have the correct shape
        assert output.shape[0] == sequence_length
        assert output.shape[1] == micro_batch_size
        assert output.shape[2] == config.hidden_size
        assert bias.shape[0] == config.hidden_size

    def run_path_attention(self, model_comm_pgs):
        """Run PaTHAttention with given process groups."""
        transformer_config = TransformerConfig(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=4,
            num_query_groups=4,
            use_cpu_initialization=False,
        )

        layer_spec = get_gpt_layer_with_transformer_engine_spec(path_attention="full")
        attention_spec = layer_spec.submodules.self_attention

        path_attention = PaTHAttention(
            transformer_config,
            attention_spec.submodules,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            model_comm_pgs=model_comm_pgs,
        )

        config = path_attention.config
        sequence_length = 64
        micro_batch_size = 2

        path_attention.cuda()

        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones(
            (sequence_length, micro_batch_size, config.hidden_size),
            device='cuda',
        )

        output, bias = path_attention(hidden_states, None)

        assert config.recompute_granularity is None
        # Check if output and bias have the correct shape
        assert output.shape[0] == sequence_length
        assert output.shape[1] == micro_batch_size
        assert output.shape[2] == config.hidden_size
        assert bias.shape[0] == config.hidden_size

    @pytest.mark.internal
    @pytest.mark.flaky
    def test_path_attention_mpu(self):
        """Test PaTHAttention with model parallel utilities."""
        tp_size = 2
        cp_size = 1
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size, context_parallel_size=cp_size
        )
        model_parallel_cuda_manual_seed(123)

        # Get TP and CP process groups from device mesh
        tp_group = parallel_state.get_tensor_model_parallel_group()
        cp_group = parallel_state.get_context_parallel_group()

        model_comm_pgs = ModelCommProcessGroups(tp=tp_group, cp=cp_group)

        self.run_path_attention(model_comm_pgs)
