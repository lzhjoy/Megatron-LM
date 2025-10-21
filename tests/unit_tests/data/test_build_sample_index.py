import sys

import numpy
import numpy as np

sys.path.append("/mnt/yulan_pretrain/pretrain-linear-moe/YuLan-Pretrain")

from megatron.core.datasets import helpers
from megatron.core.datasets.indexed_dataset import IndexedDataset


def _shuffle_fewer_truncation(
    document_index: numpy.ndarray,
    numpy_random_state: numpy.random.RandomState,
    sequence_lengths: numpy.ndarray,
    seq_length: int,
) -> numpy.ndarray:
    pad_marker = len(sequence_lengths)
    from megatron.core.datasets import helpers_cpp

    numpy_random_state.shuffle(document_index)
    res = helpers_cpp.pack_items(sequence_lengths[document_index], seq_length, mode='ffd')
    new_document_index = []
    for remaining_size, local_index in zip(res['rem'], res['bins']):
        new_document_index.extend(document_index[local_index])
        if 0 < remaining_size:
            new_document_index.extend([pad_marker] * remaining_size)
        elif remaining_size < 0:
            new_document_index.extend([pad_marker] * (seq_length - remaining_size))
    return numpy.array(new_document_index)


def _shuffle_fewer_truncation_cpp(
    document_index: numpy.ndarray,
    numpy_random_state: numpy.random.RandomState,
    sequence_lengths: numpy.ndarray,
    seq_length: int,
) -> numpy.ndarray:
    """New C++ implementation that does everything in one call"""
    pad_marker = len(sequence_lengths)
    from megatron.core.datasets import helpers_cpp

    numpy_random_state.shuffle(document_index)
    # Use the new C++ function that handles everything
    new_document_index = helpers_cpp.pack_documents(
        document_index, sequence_lengths, seq_length, mode='ffd', pad_marker=pad_marker
    )
    return new_document_index


def _build_document_index(
    documents: numpy.ndarray,
    num_epochs: int,
    numpy_random_state: numpy.random.RandomState,
    separate_final_epoch: bool,
    sequence_lengths: numpy.ndarray,
    seq_length: int
) -> numpy.ndarray:
    """Build an array with length = num epochs * num documents

    Args:
        documents (numpy.ndarray): the subset of exposed document indices

        num_epochs (int): The number of epochs

        numpy_random_state (numpy.random.RandomState): The NumPy random state

        separate_final_epoch (bool): Whether to exclude the last epoch from the global shuffle

    Returns:
        numpy.ndarray: The document index
    """
    print(f"num_epochs: {num_epochs}, separate_final_epoch: {separate_final_epoch}, documents: {documents.shape}")

    if not separate_final_epoch or num_epochs == 1:
        # iter-based dataset
        document_index = numpy.mgrid[0:num_epochs, 0:len(documents)][1]
        document_index[:] = documents
        document_index = document_index.reshape(-1)
        print(document_index.shape)
        document_index = document_index.astype(numpy.int32)
        document_index = _shuffle_fewer_truncation(document_index, numpy_random_state, sequence_lengths, seq_length)
        return document_index

    # shuffle-based dataset
    doc_idx_first = _build_document_index(documents, num_epochs - 1,
                                          numpy_random_state, False, sequence_lengths, seq_length)
    doc_idx_last = _build_document_index(documents, 1, numpy_random_state,
                                         False, sequence_lengths, seq_length)
    return numpy.concatenate((doc_idx_first, doc_idx_last))



def _get_item(idx, sizes, document_indices, sample_index, pad_marker):
    doc_index_beg, doc_index_beg_offset = sample_index[idx]
    doc_index_end, doc_index_end_offset = sample_index[idx + 1]
    if doc_index_beg == doc_index_end:
        if document_indices[doc_index_beg] != pad_marker:
            doc_indices = [document_indices[doc_index_beg]]
            num_paddings = 0
        else:
            doc_indices = []
            num_paddings = 1
        return doc_index_end_offset - doc_index_beg_offset, doc_indices, num_paddings
    else:
        sample_parts = 0
        doc_indices = []
        num_paddings = 0
        for i in range(doc_index_beg, doc_index_end):
            offset = 0 if i > doc_index_beg else doc_index_beg_offset
            length = (None if i < doc_index_end else doc_index_end_offset)
            if length is None:
                if document_indices[i] != pad_marker:
                    length = sizes[document_indices[i]] - offset
                else:
                    num_paddings += 1
                    length = 1
            if document_indices[i] != pad_marker:
                doc_indices.append(document_indices[i])
            sample_parts += length
        return sample_parts, doc_indices, num_paddings


def test_build_sample_index(num_epochs: int = 1):

    np.random.seed(42)
    seq_length = 32768
    numpy_random_state = np.random.RandomState(42)

    indexed_dataset = IndexedDataset(
        # "/mnt/yulan_pretrain/mount/data_final_train/预实验/stage_1/tmp/worker_0_input_ids_tmp",
        "/mnt/yulan_pretrain/mount/data_final_train_qwen3/test-loss-mask/stage_1/tmp/worker_0_input_ids_tmp",
        multimodal=False,
        mmap=False,
    )

    sequence_lengths = indexed_dataset.sequence_lengths
    tokens_per_epoch = np.sum(sequence_lengths)
    avg_length = np.mean(sequence_lengths)
    print(f"tokens_per_epoch: {tokens_per_epoch}")
    print(f"avg_length: {avg_length}")

    document_indices = _build_document_index(
        documents=np.arange(len(sequence_lengths)),
        num_epochs=num_epochs,
        numpy_random_state=numpy_random_state,
        separate_final_epoch=False,
        sequence_lengths=sequence_lengths,
        seq_length=seq_length,
    )
    print("document_indices:\n", document_indices)
    # print("shuffled sequence_lengths:\n", [sequence_lengths[i] for i in document_indices])

    pad_marker = len(sequence_lengths)
    sequence_lengths_for_cpp = sequence_lengths.copy()
    sequence_lengths_for_cpp = numpy.append(sequence_lengths_for_cpp, [1])
    sample_index = helpers.build_sample_idx(
        sizes=sequence_lengths_for_cpp,
        document_indices=document_indices,
        sequence_length=seq_length,
        num_epochs=num_epochs,
        tokens_per_epoch=tokens_per_epoch,
        drop_last_partial_sequence=False,
        add_extra_token_to_sequence=True,
    )
    # print("sample_index:\n", sample_index)
    total_samples = len(sample_index) - 1
    for idx in range(min(len(sample_index) - 1, 10)):
        size, docs, num_paddings = _get_item(idx, sequence_lengths_for_cpp, document_indices, sample_index, pad_marker)
        doc_length = sequence_lengths_for_cpp[docs]
        num_docs = len(docs)
        total_doc_len = doc_length.sum()
        print(f"idx: {idx}, size: {size}, docs: {docs}, num_docs: {num_docs}, doc_length: {doc_length}, total_doc_len: {total_doc_len}, sample_indx: {sample_index[idx]}, num_paddings: {num_paddings}")
    total_docs = 0
    total_paddings = 0
    total_tokens = total_samples * seq_length
    for idx in range(len(sample_index) - 1):
        _, docs, num_paddings = _get_item(idx, sequence_lengths_for_cpp, document_indices, sample_index, pad_marker)
        total_docs += len(docs)
        total_paddings += num_paddings
    print(f"total_docs: {total_docs}")
    print(f"total_paddings: {total_paddings}, total_tokens: {total_tokens}, paddings_ratio: {total_paddings / total_tokens}")
    truncations = total_docs - len(sequence_lengths) * num_epochs
    print("truncations: ", truncations)
    print("len(sequence_lengths): ", len(sequence_lengths))
    print("truncation ratio", truncations / len(sequence_lengths))

def test_cpp_vs_python_implementation():
    """Test that the new C++ implementation produces the same results as Python"""
    np.random.seed(42)
    seq_length = 1024

    # Create some test data
    sequence_lengths = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1500])  # Some oversized
    document_index = np.arange(len(sequence_lengths))

    # Test both implementations with the same random state
    numpy_random_state1 = np.random.RandomState(123)
    numpy_random_state2 = np.random.RandomState(123)

    result_python = _shuffle_fewer_truncation(
        document_index.copy(), numpy_random_state1, sequence_lengths, seq_length
    )

    result_cpp = _shuffle_fewer_truncation_cpp(
        document_index.copy(), numpy_random_state2, sequence_lengths, seq_length
    )

    print("Python result length:", len(result_python))
    print("C++ result length:", len(result_cpp))
    print("Results are equal:", np.array_equal(result_python, result_cpp))

    if not np.array_equal(result_python, result_cpp):
        print("Python result:", result_python[:50], "..." if len(result_python) > 50 else "")
        print("C++ result:", result_cpp[:50], "..." if len(result_cpp) > 50 else "")
    else:
        print("✅ Both implementations produce identical results!")


def test_helpers_cpp_pack_items():
    """Test the pack_items function in helpers_cpp"""
    from megatron.core.datasets import helpers_cpp

    # Test basic packing
    lengths = np.array([100, 200, 300, 400, 500])
    L = 1000

    result = helpers_cpp.pack_items(lengths, L, mode='ffd')
    print("pack_items result keys:", result.keys())
    print("assignments:", result['assignments'])
    print("bins:", result['bins'])
    print("rem:", result['rem'])

    # Test with oversized items
    lengths_oversized = np.array([100, 200, 1500, 300, 2000])  # 1500 and 2000 > L=1000
    result_oversized = helpers_cpp.pack_items(lengths_oversized, L, mode='ffd')
    print("\nWith oversized items:")
    print("assignments:", result_oversized['assignments'])
    print("bins:", result_oversized['bins'])
    print("rem:", result_oversized['rem'])


if __name__ == "__main__":
    test_build_sample_index(num_epochs=2)
