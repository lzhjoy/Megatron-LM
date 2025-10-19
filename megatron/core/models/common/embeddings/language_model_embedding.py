# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from typing import Literal, Optional

import torch
from torch import Tensor

from megatron.core import tensor_parallel
from megatron.core import parallel_state
from megatron.core.tensor_parallel.utils import VocabUtility
from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import get_tensor_model_parallel_group_if_none, nvtx_decorator
from megatron.core.transformer.utils import save_to_hidden_states_tracker


class EmbDeviationLossAutoScaler(torch.autograd.Function):
    """An AutoScaler that triggers the backward pass and scales the grad for emb deviation loss."""

    main_loss_backward_scale: torch.Tensor = torch.tensor(1.0)

    @staticmethod
    def forward(ctx, output: torch.Tensor, emb_deviation_loss: torch.Tensor):
        """Preserve the emb deviation by storing it in the context to avoid garbage collection.

        Args:
            output (torch.Tensor): The output tensor.
            emb_deviation_loss (torch.Tensor): The emb deviation loss tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        ctx.save_for_backward(emb_deviation_loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Compute and scale the gradient for emb deviation loss.

        Args:
            grad_output (torch.Tensor): The gradient of the output.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The gradient of the output, scaled emb deviation
                                               loss gradient.
        """
        (emb_deviation_loss,) = ctx.saved_tensors
        emb_deviation_loss_backward_scale = EmbDeviationLossAutoScaler.main_loss_backward_scale
        scaled_emb_deviation_loss_grad = torch.ones_like(emb_deviation_loss) * emb_deviation_loss_backward_scale
        return grad_output, scaled_emb_deviation_loss_grad

    @staticmethod
    def set_loss_scale(scale: torch.Tensor):
        """set the scale of the emb deviation loss.

        Args:
            scale (torch.Tensor): The scale value to set. Please ensure that the scale passed in
                                  matches the scale of the main_loss.
        """
        EmbDeviationLossAutoScaler.main_loss_backward_scale = scale


class LanguageModelEmbedding(MegatronModule):
    """Language model embeddings.

    Args:
        config (TransformerConfig): config object with all necessary configs for TransformerBlock
        vocab_size (int): vocabulary size
        max_sequence_length (int): maximum size of sequence. This
                             is used for positional embedding
        add_position_embedding (bool): Add a position embedding.
        embedding_dropout_prob (float): dropout probability for embeddings
        num_tokentypes (int): Set to 0 without binary head, and 2 with a binary head. Defaults to 0.
        scatter_to_sequence_parallel (bool): Set to False to disable scatter of embedding
            across sequence parallel region. Defaults to True.
    """

    def __init__(
        self,
        config: TransformerConfig,
        vocab_size: int,
        max_sequence_length: int,
        position_embedding_type: Literal['learned_absolute', 'rope', 'none'] = 'learned_absolute',
        num_tokentypes: int = 0,
        scatter_to_sequence_parallel: bool = True,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
        dropout_replacement_vocab_size: int = 151643,
    ):
        super().__init__(config=config)

        self.config: TransformerConfig = config
        self.vocab_size: int = vocab_size
        self.max_sequence_length: int = max_sequence_length
        self.add_position_embedding: bool = position_embedding_type == 'learned_absolute'
        self.num_tokentypes = num_tokentypes
        self.scatter_to_sequence_parallel = scatter_to_sequence_parallel
        self.tp_group = get_tensor_model_parallel_group_if_none(tp_group)
        self.reduce_scatter_embeddings = (
            (not self.add_position_embedding)
            and self.num_tokentypes <= 0
            and self.config.sequence_parallel
            and self.scatter_to_sequence_parallel
        )

        # Word embeddings (parallel).
        self.word_embeddings = tensor_parallel.VocabParallelEmbedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.config.hidden_size,
            init_method=self.config.embedding_init_method,
            reduce_scatter_embeddings=self.reduce_scatter_embeddings,
            config=self.config,
            tp_group=self.tp_group,
        )

        with torch.no_grad():
            # 1. 获取当前 GPU 上的词向量分片
            # self.word_embeddings.weight 的形状是 [local_vocab_size, hidden_size]
            local_embedding_shard = self.word_embeddings.weight

            get_vocab_range = VocabUtility.vocab_range_from_per_partition_vocab_size
            partition_vocab_size = local_embedding_shard.size()[0]
            rank = get_tensor_model_parallel_rank()
            world_size = get_tensor_model_parallel_world_size()
            start_idx, end_idx = get_vocab_range(partition_vocab_size, rank, world_size)
            # print("start_idx: ", start_idx)
            # print("end_idx: ", end_idx)

            # 只关心 0 ~ dropout_replacement_vocab_size-1
            effective_end_idx = min(end_idx, dropout_replacement_vocab_size)

            local_sum_vector = torch.zeros(
                self.config.hidden_size,
                dtype=self.config.params_dtype,
                device=local_embedding_shard.device,
            )

            if start_idx < effective_end_idx:                # 该分片与需求区间有交集
                local_slice_start = 0                        # 在本 shard 内起始位置
                local_slice_end   = effective_end_idx - start_idx
                local_sum_vector  = local_embedding_shard[local_slice_start:local_slice_end].sum(dim=0)

            torch.distributed.all_reduce(
                local_sum_vector,
                group=get_tensor_model_parallel_group()
            )

            avg_embedding = local_sum_vector / dropout_replacement_vocab_size
            self.avg_embedding_for_dropout = avg_embedding

        # Position embedding (serial).
        if self.add_position_embedding:
            self.position_embeddings = torch.nn.Embedding(
                self.max_sequence_length, self.config.hidden_size
            )

            # Initialize the position embeddings.
            if self.config.perform_initialization:
                self.config.embedding_init_method(self.position_embeddings.weight)

        if self.num_tokentypes > 0:
            self.tokentype_embeddings = torch.nn.Embedding(
                self.num_tokentypes, self.config.hidden_size
            )
            # Initialize the token-type embeddings.
            if self.config.perform_initialization:
                self.config.embedding_init_method(self.tokentype_embeddings.weight)
        else:
            self.tokentype_embeddings = None

        # Embeddings dropout
        self.embedding_dropout = torch.nn.Dropout(self.config.hidden_dropout)

    def zero_parameters(self):
        """Zero out all parameters in embedding."""
        self.word_embeddings.weight.data.fill_(0)
        self.word_embeddings.weight.shared = True
        self.position_embeddings.weight.data.fill_(0)
        self.position_embeddings.weight.shared = True
        if self.num_tokentypes > 0:
            self.tokentype_embeddings.weight.data.fill_(0)
            self.tokentype_embeddings.weight.shared = True

    @nvtx_decorator()
    def forward(self, input_ids: Tensor, position_ids: Tensor, tokentype_ids: int = None, target_dataset_mask: Tensor = None) -> Tensor:
        """Forward pass of the embedding module.

        Args:
            input_ids (Tensor): The input tokens
            position_ids (Tensor): The position id's used to calculate position embeddings
            tokentype_ids (int): The token type ids. Used when args.bert_binary_head is set to True. Defaults to None
            target_dataset_mask (Tensor): Whether to apply dropout for each token.

        Returns:
            Tensor: The output embeddings
        """
        word_embeddings = self.word_embeddings(input_ids)

        from megatron.training import get_args

        args = get_args()
        # print('args.word_embedding_dropout_prob: ', args.word_embedding_dropout_prob)

        if (self.training and
            hasattr(args, 'word_embedding_dropout_prob') and
            args.word_embedding_dropout_prob > 0.0 and
            target_dataset_mask is not None):

            # 将target_dataset_mask中的2替换为1，3替换为0
            target_dataset_mask = torch.where(target_dataset_mask == 2, 1,
                     torch.where(target_dataset_mask == 3, 0, target_dataset_mask))


            p_max = args.word_embedding_dropout_prob
            batch_size = target_dataset_mask.shape[0]

            # 为每个序列生成一个 [0, 1] 之间的随机数，然后乘以 p_max
            # dynamic_probs_per_sequence 形状: [batch_size]
            dynamic_probs_per_sequence = torch.rand(
                batch_size,
                device=target_dataset_mask.device
            ) * p_max

            # 为了广播，将其变形为 [batch_size, 1]
            dynamic_probs_per_sequence = dynamic_probs_per_sequence.unsqueeze(1)

            # 为每个 token 生成一个 [0, 1] 的随机数
            per_token_rand = torch.rand_like(target_dataset_mask, dtype=torch.float32)

            # 一个 token 被 dropout，当且仅当它的随机数小于它所在序列的动态概率
            # PyTorch 的广播机制会自动处理这里的比较
            # ( [b, s] < [b, 1] ) -> [b, s]
            dropout_mask = per_token_rand < dynamic_probs_per_sequence

            # 只有当一个 token 既属于目标数据集，又被选中 dropout 时，才最终被置零
            final_mask = (target_dataset_mask & dropout_mask).bool()

            replacement_embedding = self.avg_embedding_for_dropout.to(
                device=word_embeddings.device
            ).view(1, 1, -1)

            # 扩展掩码维度以匹配 embedding 张量
            # final_mask: [b, s] -> [b, s, 1]
            final_mask_expanded = final_mask.unsqueeze(-1)

            # 在原位(in-place)将选中的词向量置零
            word_embeddings = torch.where(
                final_mask_expanded,
                replacement_embedding,
                word_embeddings
            )


        if self.add_position_embedding:
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = word_embeddings + position_embeddings
        else:
            embeddings = word_embeddings

        if not self.reduce_scatter_embeddings:
            # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
            embeddings = embeddings.transpose(0, 1).contiguous()

        if tokentype_ids is not None:
            assert self.tokentype_embeddings is not None
            # [b s h] -> [s b h] (So that it can be added with embeddings)
            tokentype_embedding = self.tokentype_embeddings(tokentype_ids).permute(1, 0, 2)
            embeddings = embeddings + tokentype_embedding
        else:
            assert self.tokentype_embeddings is None

        # If the input flag for fp32 residual connection is set, convert for float.
        if self.config.fp32_residual_connection:
            embeddings = embeddings.float()

        # Dropout.
        if self.config.sequence_parallel:
            if not self.reduce_scatter_embeddings and self.scatter_to_sequence_parallel:
                embeddings = tensor_parallel.scatter_to_sequence_parallel_region(
                    embeddings, group=self.tp_group
                )
            # `scatter_to_sequence_parallel_region` returns a view, which prevents
            # the original tensor from being garbage collected. Clone to facilitate GC.
            # Has a small runtime cost (~0.5%).
            if self.config.clone_scatter_output_in_embedding and self.scatter_to_sequence_parallel:
                embeddings = embeddings.clone()
            with tensor_parallel.get_cuda_rng_tracker().fork():
                embeddings = self.embedding_dropout(embeddings)
        else:
            embeddings = self.embedding_dropout(embeddings)

        # Embedding deviation loss.
        if self.config.emb_deviation_type is not None:
            if self.config.emb_deviation_type == "square_loss":
                emb_deviation_loss = (embeddings.mean() ** 2) * self.config.emb_deviation_loss_coeff
            elif self.config.emb_deviation_type == "loss":
                emb_deviation_loss = torch.abs(embeddings.mean()) * self.config.emb_deviation_loss_coeff

            if self.config.emb_deviation_type == "mean":
                embeddings = embeddings - embeddings.mean(dim=-1, keepdim=True)
            else:
                if self.config.calculate_per_token_loss:
                    embeddings = EmbDeviationLossAutoScaler.apply(embeddings, emb_deviation_loss * embeddings.shape[0])
                else:
                    embeddings = EmbDeviationLossAutoScaler.apply(embeddings, emb_deviation_loss)

        # Log hidden states.
        if isinstance(self.config.log_layer_hidden_states, list) and "embeddings" in self.config.log_layer_hidden_states:
            save_to_hidden_states_tracker("embeddings", embeddings, 0, self.config.num_layers, avg_group=self.sp_group)

        return embeddings
