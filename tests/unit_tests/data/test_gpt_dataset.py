# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

##
# Compile megatron.core.datasets.helpers_cpp dependencies before BlendedDataset import
##

import os
import random
import sys

os.environ["LOCAL_RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"

import numpy
import pytest
import torch

sys.path.append("/data/pretrain-linear-moe/YuLan-Pretrain")

from megatron.core.datasets.blended_megatron_dataset_builder import \
    BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import (GPTDataset, GPTDatasetConfig,
                                                MockGPTDataset)
from megatron.core.datasets.utils import compile_helpers, get_blend_from_list
from megatron.training.tokenizer.tokenizer import _NullTokenizer
from tests.unit_tests.test_utilities import Utils

_MOCK_VOCAB_SIZE = 8192


def sample_N(dataset, N, randomize):
    if randomize:
        indices = [random.randint(0, len(dataset) - 1) for _ in range(N)]
    else:
        indices = list(range(N))
    samples = [dataset[index]["tokens"].numpy() for index in indices]
    return samples


def test_mock_gpt_dataset():
    if torch.distributed.is_available():
        Utils.initialize_distributed()
        if torch.distributed.get_rank() == 0:
            compile_helpers()
        torch.distributed.barrier()
    else:
        compile_helpers()

    tokenizer = _NullTokenizer(vocab_size=_MOCK_VOCAB_SIZE)

    config = GPTDatasetConfig(
        random_seed=1234,
        sequence_length=1024,
        split="990,9,1",
        reset_position_ids=True,
        reset_attention_mask=True,
        eod_mask_loss=True,
        tokenizer=tokenizer,
        mid_level_dataset_surplus=0.005,
    )

    datasets = BlendedMegatronDatasetBuilder(MockGPTDataset, [100, 100, 100],
                                             lambda: True, config).build()

    N = 10

    # Check iso-index variance by split
    subsets = [sample_N(dataset, N, randomize=False) for dataset in datasets]
    assert not numpy.allclose(subsets[0], subsets[1])
    assert not numpy.allclose(subsets[0], subsets[2])
    assert not numpy.allclose(subsets[1], subsets[2])

    # Check iso-split / iso-index identity
    subset_1A = sample_N(datasets[0], N, randomize=False)
    subset_1B = sample_N(datasets[0], N, randomize=False)
    assert numpy.allclose(subset_1A, subset_1B)

    # Check iso-split variance by index
    subset_1A = sample_N(datasets[0], N, randomize=True)
    subset_1B = sample_N(datasets[0], N, randomize=True)
    assert not numpy.allclose(subset_1A, subset_1B)

    config = GPTDatasetConfig(
        random_seed=1234,
        sequence_length=1024,
        split="990,10,0",
        reset_position_ids=True,
        reset_attention_mask=True,
        eod_mask_loss=True,
        drop_last_partial_validation_sequence=False,
        add_extra_token_to_sequence=False,
        tokenizer=tokenizer,
        mid_level_dataset_surplus=0.005,
    )

    datasets = BlendedMegatronDatasetBuilder(MockGPTDataset, [0, None, 0],
                                             lambda: True, config).build()

    sample = datasets[1][datasets[1].shuffle_index.argmax()]
    argmax = sample['labels'].shape[0] - torch.flip(sample['labels'],
                                                    [0]).argmax() - 1

    # Test add_extra_token_to_sequence
    assert sample['tokens'][argmax] != tokenizer.eod
    assert sample['labels'][argmax] == tokenizer.eod

    # Test eod_mask_loss, drop_last_partial_validation_sequence
    assert argmax < sample['labels'].shape[0] - 1
    assert torch.all(sample['labels'][argmax + 1:] == 0)
    assert not torch.any(sample['loss_mask'][torch.logical_and(
        sample['labels'] == tokenizer.eod, sample['labels'] == 0)])

    sample = datasets[1][None]

    # Check handling of None index
    assert not torch.any(sample['loss_mask'])


def test_gpt_dataset_with_loss_mask():
    # if torch.distributed.is_available():
    #     Utils.initialize_distributed()
    #     if torch.distributed.get_rank() == 0:
    #         compile_helpers()
    #     torch.distributed.barrier()
    # else:
    #     compile_helpers()

    data_path = []
    for i in range(16):
        d = f"/data/pretrain-linear-moe-dev/cache/datasets/IvanHU/aabcfab61cae2f6879866e11565c09b0a397c4690ee346b3135cd60c22f9d390/worker_{i}_input_ids_tmp"
        if os.path.getsize(d + ".bin") > 0:
            data_path.append("1")
            data_path.append(d)

    tokenizer = _NullTokenizer(vocab_size=_MOCK_VOCAB_SIZE)

    config = GPTDatasetConfig(
        random_seed=1234,
        sequence_length=1024,
        blend=get_blend_from_list(data_path),
        split="100,0,0",
        reset_position_ids=True,
        reset_attention_mask=True,
        eod_mask_loss=True,
        tokenizer=tokenizer,
        mid_level_dataset_surplus=0.005,
    )

    dataset, val_dataset, test_dataset = BlendedMegatronDatasetBuilder(
        GPTDataset, [100, 100, 100], lambda: True, config).build()
    print(type(dataset))

    batch = dataset[0]
    tokens = batch["tokens"].numpy()
    labels = batch["labels"].numpy()
    loss_mask = batch["loss_mask"].numpy()
    attention_mask = batch["attention_mask"].numpy()
    position_ids = batch["position_ids"].numpy()
    print(tokens, tokens.shape)
    print(labels, labels.shape)
    print(loss_mask, loss_mask.shape)
    print(attention_mask, attention_mask.shape)
    print(position_ids, position_ids.shape)
    # print(batch["dropout_mask"].numpy())

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "/data/pretrain-linear-moe/preprocess/modify_tokenizer/YuLan-Mini-Qwen-Template"
    )

    def print_text(text):
        print(text)
        print("[1]", text.count('[1]'))
        print("[2]", text.count('[2]'))
        print("[3]", text.count('[3]'))
        print("[4]", text.count('[4]'))
        print("[21]", text.count('[21]'))
        print("[22]", text.count('[22]'))
        print("[23]", text.count('[23]'))
        print("[24]", text.count('[24]'))
        print("[31]", text.count('[31]'))
        print("[32]", text.count('[32]'))
        print("[33]", text.count('[33]'))
        print("[34]", text.count('[34]'))

    text = tokenizer.decode(tokens)
    print_text(text)

    for batch in dataset:
        text = tokenizer.decode(batch["tokens"])
        print_text(text)


if __name__ == "__main__":
    # test_mock_gpt_dataset()
    test_gpt_dataset_with_loss_mask()
