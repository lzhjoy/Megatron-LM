# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Utilities for transformer layers."""
from functools import lru_cache
from operator import itemgetter
from typing import Any, Dict, Iterable, Optional, Tuple, Union, List

import torch

from megatron.core import parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedObject, ShardedStateDict, StateDict
from megatron.core.jit import jit_fuser
from megatron.core.utils import (
    make_sharded_tensor_for_checkpoint,
    make_tp_sharded_tensor_for_checkpoint,
)


# GPT logging
_GPT_LAYER_WISE_LOGGING_TRACKER = {}


def get_gpt_layer_wise_logging_tracker():
    """Return the gpt layer wise tracker."""
    global _GPT_LAYER_WISE_LOGGING_TRACKER
    return _GPT_LAYER_WISE_LOGGING_TRACKER


def save_to_hidden_states_tracker(
    name: str,
    hidden_states: torch.Tensor,
    layer_number: int,
    num_layers: int,
    reduce_group: torch.distributed.ProcessGroup = None,
    avg_group: torch.distributed.ProcessGroup = None,
):
    """Save the mean and std of hidden states for logging.
    Args:
        name (str): The name of the hidden states.
        hidden_states (torch.Tensor): The hidden states tensor.
        layer_number (int): Layer index of the loss.
        num_layers (int): The number of total layers.
        reduce_group (torch.distributed.ProcessGroup): The group for reducing the loss.
        mean_group (torch.distributed.ProcessGroup): The group for averaging the loss.
    """
    # Skip hidden states logging if layer_number is None.
    if layer_number is None:
        return

    tracker = get_gpt_layer_wise_logging_tracker()
    if name not in tracker:
        tracker[name] = {}
        tracker[name]["mean"] = torch.zeros(num_layers + 2, device=hidden_states.device)
        tracker[name]["std"] = torch.zeros(num_layers + 2, device=hidden_states.device)
        tracker[name]["num_micro_batches"] = torch.zeros(num_layers + 2, device=hidden_states.device)

    # Aggregate the values for the layer.
    d_hidden_states = hidden_states.detach()
    tracker[name]["mean"][layer_number] += d_hidden_states.mean()
    tracker[name]["std"][layer_number] += d_hidden_states.std(dim=-1).mean()
    tracker[name]["num_micro_batches"][layer_number] += 1
    tracker[name]["reduce_group"] = reduce_group
    tracker[name]["avg_group"] = avg_group


def clear_hidden_states_tracker():
    """Clear the hidden states metrics."""
    tracker = get_gpt_layer_wise_logging_tracker()
    for name in tracker:
        tracker[name]["mean"].zero_()
        tracker[name]["std"].zero_()
        tracker[name]["num_micro_batches"].zero_()
        tracker[name]["reduce_group"] = None
        tracker[name]["avg_group"] = None


def reduce_hidden_states_tracker_across_ranks(value_names: Optional[List[str]] = None, track_names: Optional[List[str]] = None):
    """Collect and reduce the hidden states across ranks."""
    tracker = get_gpt_layer_wise_logging_tracker()
    if track_names is None:
        track_names = tracker.keys()
    if value_names is None:
        value_names = ['mean', 'std']
    for name in track_names:
        for vlaue_name in value_names:
            values = tracker[name][vlaue_name]
            # TODO(Hepteract): delete the usage of the global parallel_state.
            # Collect aux losses across PP.
            torch.distributed.all_reduce(
                values, group=parallel_state.get_pipeline_model_parallel_group()
            )
            # Reduce aux losses across ranks.
            if tracker[name].get('reduce_group') is not None:
                torch.distributed.all_reduce(values, group=tracker[name].get('reduce_group'))
            if tracker[name].get('avg_group') is not None:
                torch.distributed.all_reduce(
                    values, group=tracker[name]['avg_group'], op=torch.distributed.ReduceOp.AVG
                )


def track_gpt_metrics(
    iteration: int,
    writer,
    wandb_writer=None,
    per_layer_logging=False,
    force_initialize: bool = False,
    track_names: Optional[List[str]] = None,
    num_layers: Optional[int] = None,
):
    """Track the GPT metrics for logging."""
    value_names = ["std", "mean"]

    # hidden states logging
    tracker = get_gpt_layer_wise_logging_tracker()
    # Initialize the tracker if force_initialize is True
    if force_initialize:
        if track_names is not None:
            for key in track_names:
                if key not in tracker:
                    tracker[key] = {vn: torch.zeros(num_layers + 2, device="cuda") for vn in value_names}
                    tracker[key]["reduce_group"] = None
                    tracker[key]["avg_group"] = None
    reduce_hidden_states_tracker_across_ranks(value_names, track_names)

    # only the last rank have a writer
    if writer is not None:
        value_tensors = {k: {vn: v[vn].float() for vn in value_names} for k, v in tracker.items()}
        for name, tensor_dict in value_tensors.items():

            # currently when using add_scalars,
            # torch.utils.add_scalars makes each timer its own run, which
            # polutes the runs list, so we just add each as a scalar
            total_scale = tracker[name]['num_micro_batches'].sum()
            for vn, tensor in tensor_dict.items():
                writer.add_scalar(f"gpt_{vn}/{name}", tensor.sum() / total_scale, iteration)
                if per_layer_logging:
                    for i, val in enumerate(tensor.tolist()):
                        layer_scale = tracker[name]['num_micro_batches'][i].item()
                        if layer_scale == 0:
                            continue
                        writer.add_scalar(f"gpt_{vn}/layer_{i}_{name}", val / layer_scale, iteration)

            # W&B logging lacks support for logging multiple scalars simultaneously.
            # As a workaround, we log each scalar individually first, then we can create
            # a custom panel to manually group them to a single plot.
            if wandb_writer:
                for vn, tensor in tensor_dict.items():
                    wandb_writer.log({f"gpt_{vn}/{name}": tensor.sum() / total_scale}, iteration)
                    if per_layer_logging:
                        wandb_writer.log(
                            {
                                f"gpt_{vn}/layer_{i}_{name}": loss / nmb
                                for i, (loss, nmb) in enumerate(zip(tensor.tolist(), tracker[name]['num_micro_batches'].tolist()))
                                if nmb > 0
                            },
                            iteration,
                        )

    clear_hidden_states_tracker()


def get_linear_layer(rows, columns, init_method, perform_initialization=True):
    """Simple linear layer with weight initialization."""
    layer = torch.nn.Linear(rows, columns)
    if perform_initialization:  # Take from modelparallel config
        init_method(layer.weight)
    with torch.no_grad():
        layer.bias.zero_()
    return layer


@lru_cache(maxsize=32)
def get_default_causal_mask(sq: int) -> torch.Tensor:
    """Return the causal upper triangular mask for softmax input."""
    return torch.triu(torch.ones(sq, sq, device="cuda"), diagonal=1).bool()


# pylint: disable=missing-function-docstring
def attention_mask_func(attention_scores, attention_mask):
    attention_scores.masked_fill_(attention_mask, -10000.0)
    return attention_scores


@jit_fuser
def gelu_impl(x):
    """OpenAI's gelu implementation."""
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x * x)))


# pylint: disable=missing-function-docstring
def openai_gelu(x):
    return gelu_impl(x)


# This is actually Python equivalent of torch.nn.functional.gelu(), also with
# type hints for ONNX exporter
# pylint: disable=missing-function-docstring
@jit_fuser
def erf_gelu(x):
    return (
        x * 0.5 * (torch.erf(x / 1.41421).to(dtype=x.dtype) + torch.ones_like(x).to(dtype=x.dtype))
    )


def make_sharded_tensors_for_checkpoint(
    state_dict: StateDict,
    prefix: str,
    tensor_parallel_layers_axis_map: Optional[Dict[str, int]] = None,
    sharded_offsets: Iterable[Tuple[int, int, int]] = (),
    extra_state_suffix: str = '_extra_state',
):
    """Wraps tensors from transformer layers with ShardedTensor or ShardedObject.

    For a given `state_dict`, wraps:
    - all _extra_states with ShardedObject
    - all tensors specified in tensor_parallel_layers_axis_map with TP and DP sharded ShardedTensor
    - other values with DP sharded ShardedTensor

    Args:
        state_dict (StateDict): state_dict to convert
        prefix (str): prefix appended to keys in final state dict
        tensor_parallel_layers_axis_map (Dict[str, int], optional): dict mapping layer
            names to the axis for TP sharding
        sharded_offsets (Iterable[Tuple[int, int, int]], optional): sharding already
            applied (e.g. PP related), passed along to ShardedTensor
        extra_state_suffix (str, default = '_extra_state'): layers with this
            suffix will be wrapped with ShardedObject instead of ShardedTensor.

    """

    if tensor_parallel_layers_axis_map is None:
        tensor_parallel_layers_axis_map = {}

    sharded_state_dict = {}
    for layer_name in state_dict.keys():
        tensor = state_dict[layer_name]
        layer_key = f'{prefix}{layer_name}'

        if layer_name.endswith(extra_state_suffix):
            sharded_state_dict[layer_key] = make_sharded_object_for_checkpoint(
                tensor, layer_key, sharded_offsets
            )

        elif layer_name in tensor_parallel_layers_axis_map:
            tp_axis = tensor_parallel_layers_axis_map[layer_name]
            sharded_state_dict[layer_key] = make_tp_sharded_tensor_for_checkpoint(
                tensor, layer_key, tp_axis, prepend_offsets=sharded_offsets
            )

        else:
            sharded_state_dict[layer_key] = make_sharded_tensor_for_checkpoint(
                tensor, layer_key, prepend_offsets=sharded_offsets
            )

    return sharded_state_dict


def make_sharded_object_for_checkpoint(
    obj: Any,
    key: str,
    sharded_offsets: Iterable[Tuple[int, int, int]] = (),
    replica_id: Union[None, int, Tuple[int, ...]] = None,
    **kwargs,
):
    """Helper for instantiating a non-sharded ShardedObject (replicated across TP and DP group).

    Args:
        obj (object): any object to be sharded
        key (str): unique identifier of the object
        sharded_offsets (Iterable[Tuple[int, int, int]]): offsets normally
            prepended to ShardedTensors, will be used as global offsets for
            ShardedObject
        replica_id (Union[None, int, Tuple[int, ...]]): replica id
    """
    is_obj_fully_sharded = hasattr(obj, 'fully_shard_param_local_index')
    assert not is_obj_fully_sharded, f"Fully sharded object not supported: {key}"

    if replica_id is None:
        replica_id = (
            0,
            parallel_state.get_tensor_model_parallel_rank(),
            parallel_state.get_data_parallel_rank(with_context_parallel=True),
        )

    return ShardedObject(key, obj, *_get_extra_state_offsets(sharded_offsets), replica_id, **kwargs)


def _get_extra_state_offsets(
    sharded_offsets: Iterable[Tuple[int, int, int]]
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Turns ShardedTensor offsets into offsets suitable for ShardedObject."""
    if sharded_offsets:
        sharded_offsets = sorted(sharded_offsets, key=itemgetter(0))  # sort by axis
        axis, extra_state_offset, extra_state_shape = zip(*sharded_offsets)
        assert list(axis) == list(
            range(len(axis))
        ), f'Expected contiguous axis for offsets: {sharded_offsets}'
    else:
        extra_state_shape = (1,)
        extra_state_offset = (0,)
    return extra_state_shape, extra_state_offset


def sharded_state_dict_default(
    module: torch.nn.Module,
    prefix: str = '',
    sharded_offsets: Tuple[Tuple[int, int, int]] = (),
    metadata: Optional[dict] = None,
) -> ShardedStateDict:
    """Provides implementation for sharded_state_dict method for non-MegatronModules.

    Tries to call `module.sharded_state_dict` when possible,
    otherwise uses regular state dict and assumes tensors are replicated across TP and DP.

    `keep_vars=True` is passed to module.state_dict so that optimizer states
    can be sharded later on.

    Args:
        module (torch.nn.Module): module which sharded state dict we want to obtain
        prefix (str): prefix for the state dict keys
        sharded_offsets (Tuple[Tuple[int, int, int]], optional): sharding already
            applied (e.g. PP related) by sup-modules. Passed along to ShardedTensor
        metadata (dict, optional): metadata passed to module sharded_state_dict method

    Returns:
        dict: dictionary of state dict keys mapped to ShardedTensors
    """

    if hasattr(module, 'sharded_state_dict'):
        module_sharded_sd = module.sharded_state_dict(
            prefix=prefix, sharded_offsets=sharded_offsets, metadata=metadata
        )
    else:
        module_sd = module.state_dict(prefix='', keep_vars=True)
        module_sharded_sd = make_sharded_tensors_for_checkpoint(
            module_sd, prefix, {}, sharded_offsets
        )
    return module_sharded_sd


# Initialize cache for sequence parallel modules
_sequence_parallel_attr_cache = None


def _init_sequence_parallel_cache(model):
    """
    Initialize the cache of modules with sequence parallel attributes.
    Only needs to be called once, subsequent calls have no effect.
    """
    global _sequence_parallel_attr_cache
    model_id = id(model)
    if _sequence_parallel_attr_cache is not None and model_id in _sequence_parallel_attr_cache:
        return  # Cache already initialized

    # Attributes for sequence parallel
    sequence_parallel_attrs = [
        "sequence_parallel",
        "scatter_to_sequence_parallel",
        "reduce_scatter_embeddings",
    ]

    if model.position_embedding_type == "learned_absolute":
        sequence_parallel_attrs.remove("reduce_scatter_embeddings")

    # Initialize dictionary to hold attributes -> list of modules
    if _sequence_parallel_attr_cache is None:
        _sequence_parallel_attr_cache = {}
    _sequence_parallel_attr_cache[model_id] = {attr: [] for attr in sequence_parallel_attrs}

    # Get the model
    model_modules = model

    # Recursive function to find all modules with our target attributes
    def find_modules_with_attrs(module):
        # Check if this module has any of our target attributes
        for attr in sequence_parallel_attrs:
            if hasattr(module, attr):
                _sequence_parallel_attr_cache[model_id][attr].append(module)

        # Check all children modules recursively
        for child in module._modules.values():
            if child is not None:
                find_modules_with_attrs(child)

    # Start the search from each major component
    find_modules_with_attrs(model_modules)


def set_model_to_sequence_parallel(model, set_to=False):
    """
    Set sequence parallel attributes for the model.

    Args:
        set_to: Value to set for sequence_parallel attributes
    """
    global _sequence_parallel_attr_cache
    model_id = id(model)

    # Initialize cache if needed
    if _sequence_parallel_attr_cache is None or model_id not in _sequence_parallel_attr_cache:
        _init_sequence_parallel_cache(model)

    model.config.sequence_parallel = set_to

    # Set all cached attributes to desired value
    for attr, modules in _sequence_parallel_attr_cache[model_id].items():
        for module in modules:
            setattr(module, attr, set_to)
