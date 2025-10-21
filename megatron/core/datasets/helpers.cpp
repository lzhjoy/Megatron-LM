/* Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved. */

/* Helper methods for fast index mapping builds */

#include <algorithm>
#include <cassert>
#include <iostream>
#include <limits>
#include <math.h>
#include <set>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <random>

namespace py = pybind11;
using namespace std;

const int32_t LONG_SENTENCE_LEN = 512;

// Segment tree supporting "max leaf value" query where leaves are capacities 1..L
struct SegmentTree {
    int n; // number of leaves (L)
    int size; // power-of-two size
    std::vector<int> tree; // max values

    SegmentTree(int L=0) { build(L); }

    void build(int L) {
        n = L;
        size = 1;
        while (size < n+1) size <<= 1; // we index leaves by capacity value (1..n)
        tree.assign(2*size, 0);
    }

    // set leaf at position pos (1..n) to value val
    void set_leaf(int pos, int val) {
        if (pos < 1 || pos > n) return;
        int idx = size + pos;
        tree[idx] = val;
        idx >>= 1;
        while (idx) {
            tree[idx] = std::max(tree[idx<<1], tree[(idx<<1)|1]);
            idx >>= 1;
        }
    }

    // find a leaf whose value >= need. This implements the paper's query: at internal node go left if left child >= need else go right.
    // Returns capacity (leaf index) or 0 if none.
    int find_best_fit(int need) const {
        if (need <= 0) return 0;
        if (tree[1] < need) return 0; // no leaf qualifies
        int idx = 1;
        while (idx < size) {
            int left = idx<<1;
            if (tree[left] >= need) idx = left;
            else idx = left|1;
        }
        int leaf_pos = idx - size;
        if (leaf_pos >= 1 && leaf_pos <= n) return leaf_pos;
        return 0;
    }
};


void build_exhaustive_blending_indices(py::array_t<int16_t> &dataset_index, py::array_t<int64_t> &dataset_sample_index, const py::array_t<int64_t> &sizes, const int32_t num_datasets) {
  /*
      Build blending indices by sampling exactly as many samples from dataset[i]
      as is requested by sizes[i] for all i in the range [0, num_datasets).
  */
  auto dataset_index_ptr = dataset_index.mutable_unchecked<1>();
  auto dataset_sample_index_ptr = dataset_sample_index.mutable_unchecked<1>();
  auto sizes_ptr = sizes.unchecked<1>();

  int64_t total_size = 0;
  int64_t dataset_sample_counts[num_datasets];
  std::set<int32_t> dataset_unspent_indices;
  for (int32_t i = 0; i < num_datasets; ++i) {
    total_size += sizes_ptr[i];
    dataset_sample_counts[i] = 0;
    dataset_unspent_indices.insert(i);
  }

  // still need fractional weights to sample in proportion to sizes
  double weights[num_datasets];
  for (int32_t i = 0; i < num_datasets; ++i) {
    weights[i] = sizes_ptr[i] / static_cast<double>(total_size);
  }

  int64_t index_sample = 0;
  while (dataset_unspent_indices.size() > 0) {
    double index_sample_double = std::max(static_cast<double>(index_sample), 1.0);

    int64_t error_argmax = -1;
    double error_max = std::numeric_limits<double>::lowest();

    for (int32_t index_dataset : dataset_unspent_indices) {
      double error = weights[index_dataset] * index_sample_double - static_cast<double>(dataset_sample_counts[index_dataset]);
      if (error > error_max) {
        error_argmax = index_dataset;
        error_max = error;
      }
    }
    assert(error_argmax >= 0);

    // Populate the indices.
    dataset_index_ptr[index_sample] = static_cast<int16_t>(error_argmax);
    dataset_sample_index_ptr[index_sample] = dataset_sample_counts[error_argmax];

    // Update the total samples.
    dataset_sample_counts[error_argmax] += 1;

    if (sizes_ptr[error_argmax] - static_cast<double>(dataset_sample_counts[error_argmax]) == 0) {
      dataset_unspent_indices.erase(error_argmax);
    }

    index_sample += 1;
  }
}

void build_blending_indices(py::array_t<int16_t> &dataset_index,
                            py::array_t<int64_t> &dataset_sample_index,
                            const py::array_t<double> &weights,
                            const int32_t num_datasets,
                            const int64_t size, const bool verbose)
{
  /* Given multiple datasets and a weighting array, build samples
   such that it follows those wieghts.*/

  if (verbose)
  {
    std::cout << "> building indices for blended datasets ..." << std::endl;
  }

  // Get the pointer access without the checks.
  auto dataset_index_ptr = dataset_index.mutable_unchecked<1>();
  auto dataset_sample_index_ptr = dataset_sample_index.mutable_unchecked<1>();
  auto weights_ptr = weights.unchecked<1>();

  // Initialize buffer for number of samples used for each dataset.
  int64_t current_samples[num_datasets];
  for (int64_t i = 0; i < num_datasets; ++i)
  {
    current_samples[i] = 0;
  }

  // For each sample:
  for (int64_t sample_idx = 0; sample_idx < size; ++sample_idx)
  {

    // Determine where the max error in sampling is happening.
    auto sample_idx_double = std::max(static_cast<double>(sample_idx), 1.0);
    int64_t max_error_index = 0;
    double max_error = weights_ptr[0] * sample_idx_double -
                       static_cast<double>(current_samples[0]);
    for (int64_t dataset_idx = 1; dataset_idx < num_datasets; ++dataset_idx)
    {
      double error = weights_ptr[dataset_idx] * sample_idx_double -
                     static_cast<double>(current_samples[dataset_idx]);
      if (error > max_error)
      {
        max_error = error;
        max_error_index = dataset_idx;
      }
    }

    // Populate the indices.
    dataset_index_ptr[sample_idx] = static_cast<int16_t>(max_error_index);
    dataset_sample_index_ptr[sample_idx] = current_samples[max_error_index];

    // Update the total samples.
    current_samples[max_error_index] += 1;
  }

  // print info
  if (verbose)
  {
    std::cout << " > sample ratios:" << std::endl;
    for (int64_t dataset_idx = 0; dataset_idx < num_datasets; ++dataset_idx)
    {
      auto ratio = static_cast<double>(current_samples[dataset_idx]) /
                   static_cast<double>(size);
      std::cout << "   dataset " << dataset_idx << ", input: " << weights_ptr[dataset_idx] << ", achieved: " << ratio << std::endl;
    }
  }
}

template <typename T>
py::array_t<T> build_sample_idx(
  const py::array_t<int32_t> &sizes_,
  const py::array_t<int32_t> &document_idx_,
  const int32_t seq_length,
  const int32_t num_epochs,
  const int64_t tokens_per_epoch,
  const bool drop_last_partial_sequence = true,
  const int add_extra_token_to_sequence = 1
){
  /*
      Sample index (sample_idx) is used for gpt2 like dataset for which the documents are flattened
      and the samples are built based on this 1-D flatten array. It is a 2D array with sizes
      [number-of-samples + 1, 2] where [..., 0] contains the index into `doc_idx` and [..., 1] is
      the starting offset in that document.
  */

  // Consistency checks.
  assert(seq_length > 1);
  assert(num_epochs > 0);
  assert(tokens_per_epoch > 1);

  // Remove bound checks.
  auto sizes = sizes_.unchecked<1>();
  auto document_idx = document_idx_.unchecked<1>();

  // Build the sample idx as a contiguous 1-D array of type T.
  int64_t num_samples = 0;
  if (drop_last_partial_sequence == true) {
    num_samples = (num_epochs * tokens_per_epoch - add_extra_token_to_sequence) / seq_length;
  }
  else {
    num_samples = ceil(float(num_epochs * tokens_per_epoch - add_extra_token_to_sequence) / seq_length);
  }
  T *sample_idx = new T[2 * (num_samples + 1)];

  // Index into sample_idx.
  int64_t sample_idx_index = 0;
  // Index into document_idx.
  T document_idx_index = 0;
  // Begining offset for each document.
  T doc_offset = 0;
  // Start with first document and no offset.
  sample_idx[2 * sample_idx_index] = document_idx_index;
  sample_idx[2 * sample_idx_index + 1] = doc_offset;
  ++sample_idx_index;

  while (sample_idx_index <= num_samples)
  {
    // Start with a fresh sequence.
    int32_t remaining_seq_length = seq_length + add_extra_token_to_sequence;
    while (remaining_seq_length != 0)
    {
      // Get the document length.
      auto document_index = document_idx[document_idx_index];
      auto document_length = sizes[document_index] - doc_offset;
      // And add it to the current sequence.
      remaining_seq_length -= document_length;
      // If we have more than a full sequence, adjust offset and set
      // remaining length to zero so we return from the while loop.
      // Note that -1 here is for the same reason we have -1 in
      // `_num_epochs` calculations.
      if (remaining_seq_length <= 0)
      {
        doc_offset += (remaining_seq_length + document_length - add_extra_token_to_sequence);
        remaining_seq_length = 0;
      }
      else
      {
        // Otherwise, start from the begining of the next document.
        if (document_idx_index == (document_idx_.shape(0) - 1))
        {
          // If we have reached the end of the documents, break.
          assert(sample_idx_index == num_samples);
          doc_offset = sizes[document_idx[document_idx_index]] - add_extra_token_to_sequence;
          break;
        }
        ++document_idx_index;
        doc_offset = 0;
      }
    }
    // Record the sequence.
    sample_idx[2 * sample_idx_index] = document_idx_index;
    sample_idx[2 * sample_idx_index + 1] = doc_offset;
    ++sample_idx_index;
  }

  // Method to deallocate memory.
  py::capsule free_when_done(
    sample_idx,
    [](void *mem_){
	    T *mem = reinterpret_cast<T*>(mem_);
	    delete[] mem;
    }
  );

  // Return the numpy array.
  const auto byte_size = sizeof(T);
  return py::array_t<T>(
    std::vector<int64_t>{num_samples + 1, 2}, // shape
    {2 * byte_size, byte_size},               // C-style contiguous strides
    sample_idx,                               // the data pointer
    free_when_done                            // numpy array references
  );
}

inline int32_t get_target_sample_len(const int32_t short_seq_ratio,
                                     const int32_t max_length,
                                     std::mt19937 &rand32_gen)
{
  /* Training sample length. */
  if (short_seq_ratio == 0)
  {
    return max_length;
  }
  const auto random_number = rand32_gen();
  if ((random_number % short_seq_ratio) == 0)
  {
    return 2 + random_number % (max_length - 1);
  }
  return max_length;
}

template <typename DocIdx>
py::array build_mapping_impl(const py::array_t<int64_t> &docs_,
                             const py::array_t<int32_t> &sizes_,
                             const int32_t num_epochs,
                             const uint64_t max_num_samples,
                             const int32_t max_seq_length,
                             const double short_seq_prob,
                             const int32_t seed,
                             const bool verbose,
                             const int32_t min_num_sent)
{
  /* Build a mapping of (start-index, end-index, sequence-length) where
     start and end index are the indices of the sentences in the sample
     and sequence-length is the target sequence length.
  */

  // Consistency checks.
  assert(num_epochs > 0);
  assert(max_seq_length > 1);
  assert(short_seq_prob >= 0.0);
  assert(short_seq_prob <= 1.0);
  assert(seed > 0);

  // Remove bound checks.
  auto docs = docs_.unchecked<1>();
  auto sizes = sizes_.unchecked<1>();

  // For efficiency, convert probability to ratio. Note: rand() generates int.
  int32_t short_seq_ratio = 0;
  if (short_seq_prob > 0)
  {
    short_seq_ratio = static_cast<int32_t>(round(1.0 / short_seq_prob));
  }

  if (verbose)
  {
    const auto sent_start_index = docs[0];
    const auto sent_end_index = docs[docs_.shape(0) - 1];
    const auto num_sentences = sent_end_index - sent_start_index;
    cout << "    using:" << endl
         << std::flush;
    cout << "     number of documents:            " << docs_.shape(0) - 1 << endl
         << std::flush;
    cout << "     sentences range:                [" << sent_start_index << ", " << sent_end_index << ")" << endl
         << std::flush;
    cout << "     total number of sentences:      " << num_sentences << endl
         << std::flush;
    cout << "     number of epochs:               " << num_epochs << endl
         << std::flush;
    cout << "     maximum number of samples:      " << max_num_samples << endl
         << std::flush;
    cout << "     maximum sequence length:        " << max_seq_length << endl
         << std::flush;
    cout << "     short sequence probability:     " << short_seq_prob << endl
         << std::flush;
    cout << "     short sequence ration (1/prob): " << short_seq_ratio << endl
         << std::flush;
    cout << "     seed:                           " << seed << endl
         << std::flush;
  }

  // Mapping and it's length (1D).
  int64_t num_samples = -1;
  DocIdx *maps = NULL;

  // Perform two iterations, in the first iteration get the size
  // and allocate memory and in the second iteration populate the map.
  bool second = false;
  for (int32_t iteration = 0; iteration < 2; ++iteration)
  {

    // Set the seed so both iterations produce the same results.
    std::mt19937 rand32_gen(seed);

    // Set the flag on second iteration.
    second = (iteration == 1);

    // Counters:
    uint64_t empty_docs = 0;
    uint64_t one_sent_docs = 0;
    uint64_t long_sent_docs = 0;

    // Current map index.
    uint64_t map_index = 0;

    // For each epoch:
    for (int32_t epoch = 0; epoch < num_epochs; ++epoch)
    {
      if (map_index >= max_num_samples)
      {
        if (verbose && (!second))
        {
          cout << "    reached " << max_num_samples << " samples after "
               << epoch << " epochs ..." << endl
               << std::flush;
        }
        break;
      }
      // For each document:
      for (int32_t doc = 0; doc < (docs.shape(0) - 1); ++doc)
      {

        // Document sentences are in [sent_index_first, sent_index_last)
        const auto sent_index_first = docs[doc];
        const auto sent_index_last = docs[doc + 1];

        // At the begining of the document previous index is the
        // start index.
        auto prev_start_index = sent_index_first;

        // Remaining documents.
        auto num_remain_sent = sent_index_last - sent_index_first;

        // Some bookkeeping
        if ((epoch == 0) && (!second))
        {
          if (num_remain_sent == 0)
          {
            ++empty_docs;
          }
          if (num_remain_sent == 1)
          {
            ++one_sent_docs;
          }
        }

        // Detect documents with long sentences.
        bool contains_long_sentence = false;
        if (num_remain_sent > 1)
        {
          for (auto sent_index = sent_index_first;
               sent_index < sent_index_last; ++sent_index)
          {
            if (sizes[sent_index] > LONG_SENTENCE_LEN)
            {
              if ((epoch == 0) && (!second))
              {
                ++long_sent_docs;
              }
              contains_long_sentence = true;
              break;
            }
          }
        }

        // If we have more than two sentences.
        if ((num_remain_sent >= min_num_sent) && (!contains_long_sentence))
        {

          // Set values.
          auto seq_len = int32_t{0};
          auto num_sent = int32_t{0};
          auto target_seq_len = get_target_sample_len(short_seq_ratio,
                                                      max_seq_length,
                                                      rand32_gen);

          // Loop through sentences.
          for (auto sent_index = sent_index_first;
               sent_index < sent_index_last; ++sent_index)
          {

            // Add the size and number of sentences.
            seq_len += sizes[sent_index];
            ++num_sent;
            --num_remain_sent;

            // If we have reached the target length.
            // and if not only one sentence is left in the document.
            // and if we have at least two sentneces.
            // and if we have reached end of the document.
            if (((seq_len >= target_seq_len) &&
                 (num_remain_sent > 1) &&
                 (num_sent >= min_num_sent)) ||
                (num_remain_sent == 0))
            {

              // Check for overflow.
              if ((3 * map_index + 2) >
                  std::numeric_limits<int64_t>::max())
              {
                cout << "number of samples exceeded maximum "
                     << "allowed by type int64: "
                     << std::numeric_limits<int64_t>::max()
                     << endl;
                throw std::overflow_error("Number of samples");
              }

              // Populate the map.
              if (second)
              {
                const auto map_index_0 = 3 * map_index;
                maps[map_index_0] = static_cast<DocIdx>(prev_start_index);
                maps[map_index_0 + 1] = static_cast<DocIdx>(sent_index + 1);
                maps[map_index_0 + 2] = static_cast<DocIdx>(target_seq_len);
              }

              // Update indices / counters.
              ++map_index;
              prev_start_index = sent_index + 1;
              target_seq_len = get_target_sample_len(short_seq_ratio,
                                                     max_seq_length,
                                                     rand32_gen);
              seq_len = 0;
              num_sent = 0;
            }

          } // for (auto sent_index=sent_index_first; ...
        }   // if (num_remain_sent > 1) {
      }     // for (int doc=0; doc < num_docs; ++doc) {
    }       // for (int epoch=0; epoch < num_epochs; ++epoch) {

    if (!second)
    {
      if (verbose)
      {
        cout << "   number of empty documents: " << empty_docs << endl
             << std::flush;
        cout << "   number of documents with one sentence: " << one_sent_docs << endl
             << std::flush;
        cout << "   number of documents with long sentences: " << long_sent_docs << endl
             << std::flush;
        cout << "   will create mapping for " << map_index << " samples" << endl
             << std::flush;
      }
      assert(maps == NULL);
      assert(num_samples < 0);
      maps = new DocIdx[3 * map_index];
      num_samples = static_cast<int64_t>(map_index);
    }

  } // for (int iteration=0; iteration < 2; ++iteration) {

  // Shuffle.
  // We need a 64 bit random number generator as we might have more
  // than 2 billion samples.
  std::mt19937_64 rand64_gen(seed + 1);
  for (auto i = (num_samples - 1); i > 0; --i)
  {
    const auto j = static_cast<int64_t>(rand64_gen() % (i + 1));
    const auto i0 = 3 * i;
    const auto j0 = 3 * j;
    // Swap values.
    swap(maps[i0], maps[j0]);
    swap(maps[i0 + 1], maps[j0 + 1]);
    swap(maps[i0 + 2], maps[j0 + 2]);
  }

  // Method to deallocate memory.
  py::capsule free_when_done(maps, [](void *mem_)
                             {
            DocIdx *mem = reinterpret_cast<DocIdx*>(mem_);
	    delete[] mem; });

  // Return the numpy array.
  const auto byte_size = sizeof(DocIdx);
  return py::array(std::vector<int64_t>{num_samples, 3}, // shape
                   {3 * byte_size, byte_size},           // C-style contiguous strides
                   maps,                                 // the data pointer
                   free_when_done);                      // numpy array references
}

py::array build_mapping(const py::array_t<int64_t> &docs_,
                        const py::array_t<int> &sizes_,
                        const int num_epochs,
                        const uint64_t max_num_samples,
                        const int max_seq_length,
                        const double short_seq_prob,
                        const int seed,
                        const bool verbose,
                        const int32_t min_num_sent)
{

  if (sizes_.size() > std::numeric_limits<uint32_t>::max())
  {
    if (verbose)
    {
      cout << "    using uint64 for data mapping..." << endl
           << std::flush;
    }
    return build_mapping_impl<uint64_t>(docs_, sizes_, num_epochs,
                                        max_num_samples, max_seq_length,
                                        short_seq_prob, seed, verbose,
                                        min_num_sent);
  }
  else
  {
    if (verbose)
    {
      cout << "    using uint32 for data mapping..." << endl
           << std::flush;
    }
    return build_mapping_impl<uint32_t>(docs_, sizes_, num_epochs,
                                        max_num_samples, max_seq_length,
                                        short_seq_prob, seed, verbose,
                                        min_num_sent);
  }
}

template <typename DocIdx>
py::array build_blocks_mapping_impl(const py::array_t<int64_t> &docs_,
                                    const py::array_t<int32_t> &sizes_,
                                    const py::array_t<int32_t> &titles_sizes_,
                                    const int32_t num_epochs,
                                    const uint64_t max_num_samples,
                                    const int32_t max_seq_length,
                                    const int32_t seed,
                                    const bool verbose,
                                    const bool use_one_sent_blocks)
{
  /* Build a mapping of (start-index, end-index, sequence-length) where
     start and end index are the indices of the sentences in the sample
     and sequence-length is the target sequence length.
  */

  // Consistency checks.
  assert(num_epochs > 0);
  assert(max_seq_length > 1);
  assert(seed > 0);

  // Remove bound checks.
  auto docs = docs_.unchecked<1>();
  auto sizes = sizes_.unchecked<1>();
  auto titles_sizes = titles_sizes_.unchecked<1>();

  if (verbose)
  {
    const auto sent_start_index = docs[0];
    const auto sent_end_index = docs[docs_.shape(0) - 1];
    const auto num_sentences = sent_end_index - sent_start_index;
    cout << "    using:" << endl
         << std::flush;
    cout << "     number of documents:            " << docs_.shape(0) - 1 << endl
         << std::flush;
    cout << "     sentences range:                [" << sent_start_index << ", " << sent_end_index << ")" << endl
         << std::flush;
    cout << "     total number of sentences:      " << num_sentences << endl
         << std::flush;
    cout << "     number of epochs:               " << num_epochs << endl
         << std::flush;
    cout << "     maximum number of samples:      " << max_num_samples << endl
         << std::flush;
    cout << "     maximum sequence length:        " << max_seq_length << endl
         << std::flush;
    cout << "     seed:                           " << seed << endl
         << std::flush;
  }

  // Mapping and its length (1D).
  int64_t num_samples = -1;
  DocIdx *maps = NULL;

  // Acceptable number of sentences per block.
  int min_num_sent = 2;
  if (use_one_sent_blocks)
  {
    min_num_sent = 1;
  }

  // Perform two iterations, in the first iteration get the size
  // and allocate memory and in the second iteration populate the map.
  bool second = false;
  for (int32_t iteration = 0; iteration < 2; ++iteration)
  {

    // Set the flag on second iteration.
    second = (iteration == 1);

    // Current map index.
    uint64_t map_index = 0;

    uint64_t empty_docs = 0;
    uint64_t one_sent_docs = 0;
    uint64_t long_sent_docs = 0;
    // For each epoch:
    for (int32_t epoch = 0; epoch < num_epochs; ++epoch)
    {
      // assign every block a unique id
      int32_t block_id = 0;

      if (map_index >= max_num_samples)
      {
        if (verbose && (!second))
        {
          cout << "    reached " << max_num_samples << " samples after "
               << epoch << " epochs ..." << endl
               << std::flush;
        }
        break;
      }
      // For each document:
      for (int32_t doc = 0; doc < (docs.shape(0) - 1); ++doc)
      {

        // Document sentences are in [sent_index_first, sent_index_last)
        const auto sent_index_first = docs[doc];
        const auto sent_index_last = docs[doc + 1];
        const auto target_seq_len = max_seq_length - titles_sizes[doc];

        // At the begining of the document previous index is the
        // start index.
        auto prev_start_index = sent_index_first;

        // Remaining documents.
        auto num_remain_sent = sent_index_last - sent_index_first;

        // Some bookkeeping
        if ((epoch == 0) && (!second))
        {
          if (num_remain_sent == 0)
          {
            ++empty_docs;
          }
          if (num_remain_sent == 1)
          {
            ++one_sent_docs;
          }
        }
        // Detect documents with long sentences.
        bool contains_long_sentence = false;
        if (num_remain_sent >= min_num_sent)
        {
          for (auto sent_index = sent_index_first;
               sent_index < sent_index_last; ++sent_index)
          {
            if (sizes[sent_index] > LONG_SENTENCE_LEN)
            {
              if ((epoch == 0) && (!second))
              {
                ++long_sent_docs;
              }
              contains_long_sentence = true;
              break;
            }
          }
        }
        // If we have enough sentences and no long sentences.
        if ((num_remain_sent >= min_num_sent) && (!contains_long_sentence))
        {

          // Set values.
          auto seq_len = int32_t{0};
          auto num_sent = int32_t{0};

          // Loop through sentences.
          for (auto sent_index = sent_index_first;
               sent_index < sent_index_last; ++sent_index)
          {

            // Add the size and number of sentences.
            seq_len += sizes[sent_index];
            ++num_sent;
            --num_remain_sent;

            // If we have reached the target length.
            // and there are an acceptable number of sentences left
            // and if we have at least the minimum number of sentences.
            // or if we have reached end of the document.
            if (((seq_len >= target_seq_len) &&
                 (num_remain_sent >= min_num_sent) &&
                 (num_sent >= min_num_sent)) ||
                (num_remain_sent == 0))
            {

              // Populate the map.
              if (second)
              {
                const auto map_index_0 = 4 * map_index;
                // Each sample has 4 items: the starting sentence index, ending sentence index,
                // the index of the document from which the block comes (used for fetching titles)
                // and the unique id of the block (used for creating block indexes)

                maps[map_index_0] = static_cast<DocIdx>(prev_start_index);
                maps[map_index_0 + 1] = static_cast<DocIdx>(sent_index + 1);
                maps[map_index_0 + 2] = static_cast<DocIdx>(doc);
                maps[map_index_0 + 3] = static_cast<DocIdx>(block_id);
              }

              // Update indices / counters.
              ++map_index;
              ++block_id;
              prev_start_index = sent_index + 1;
              seq_len = 0;
              num_sent = 0;
            }
          } // for (auto sent_index=sent_index_first; ...
        }   // if (num_remain_sent > 1) {
      }     // for (int doc=0; doc < num_docs; ++doc) {
    }       // for (int epoch=0; epoch < num_epochs; ++epoch) {

    if (!second)
    {
      if (verbose)
      {
        cout << "   number of empty documents: " << empty_docs << endl
             << std::flush;
        cout << "   number of documents with one sentence: " << one_sent_docs << endl
             << std::flush;
        cout << "   number of documents with long sentences: " << long_sent_docs << endl
             << std::flush;
        cout << "   will create mapping for " << map_index << " samples" << endl
             << std::flush;
      }
      assert(maps == NULL);
      assert(num_samples < 0);
      maps = new DocIdx[4 * map_index];
      num_samples = static_cast<int64_t>(map_index);
    }

  } // for (int iteration=0; iteration < 2; ++iteration) {

  // Shuffle.
  // We need a 64 bit random number generator as we might have more
  // than 2 billion samples.
  std::mt19937_64 rand64_gen(seed + 1);
  for (auto i = (num_samples - 1); i > 0; --i)
  {
    const auto j = static_cast<int64_t>(rand64_gen() % (i + 1));
    const auto i0 = 4 * i;
    const auto j0 = 4 * j;
    // Swap values.
    swap(maps[i0], maps[j0]);
    swap(maps[i0 + 1], maps[j0 + 1]);
    swap(maps[i0 + 2], maps[j0 + 2]);
    swap(maps[i0 + 3], maps[j0 + 3]);
  }

  // Method to deallocate memory.
  py::capsule free_when_done(maps, [](void *mem_)
                             {
            DocIdx *mem = reinterpret_cast<DocIdx*>(mem_);
	    delete[] mem; });

  // Return the numpy array.
  const auto byte_size = sizeof(DocIdx);
  return py::array(std::vector<int64_t>{num_samples, 4}, // shape
                   {4 * byte_size, byte_size},           // C-style contiguous strides
                   maps,                                 // the data pointer
                   free_when_done);                      // numpy array references
}

py::array build_blocks_mapping(const py::array_t<int64_t> &docs_,
                               const py::array_t<int> &sizes_,
                               const py::array_t<int> &titles_sizes_,
                               const int num_epochs,
                               const uint64_t max_num_samples,
                               const int max_seq_length,
                               const int seed,
                               const bool verbose,
                               const bool use_one_sent_blocks)
{

  if (sizes_.size() > std::numeric_limits<uint32_t>::max())
  {
    if (verbose)
    {
      cout << "    using uint64 for data mapping..." << endl
           << std::flush;
    }
    return build_blocks_mapping_impl<uint64_t>(docs_, sizes_, titles_sizes_,
                                               num_epochs, max_num_samples, max_seq_length, seed, verbose, use_one_sent_blocks);
  }
  else
  {
    if (verbose)
    {
      cout << "    using uint32 for data mapping..." << endl
           << std::flush;
    }
    return build_blocks_mapping_impl<uint32_t>(docs_, sizes_, titles_sizes_,
                                               num_epochs, max_num_samples, max_seq_length, seed, verbose, use_one_sent_blocks);
  }
}

// Optimized Best-Fit-Decreasing (OBFD) packing
// Inputs:
// - lengths: 1D integer numpy array of item sizes
// - L: bin capacity (max sequence length)
// - mode: "bfd" or "ffd"
// Returns a dict: {
//   "assignments": numpy array of bin id for each input item in the original order,
//   "bins": list of lists of item indices assigned to each bin (bin ids are 0..m-1),
//   "rem": list of remaining capacities per bin
// }

py::dict pack_items(py::array_t<int, py::array::c_style | py::array::forcecast> lengths,
                    int L,
                    std::string mode = "bfd") {
    if (L <= 0) throw std::invalid_argument("L must be > 0");
    auto buf = lengths.unchecked<1>();
    ssize_t N = buf.shape(0);
    // Create items vector of pairs (length, original_index) for normal-sized items
    // and handle oversized items separately
    std::vector<std::pair<int,int>> items;
    std::vector<std::pair<int,int>> oversized_items;
    items.reserve(N);
    for (ssize_t i = 0; i < N; ++i) {
        int v = buf(i);
        if (v <= 0) {
            throw std::invalid_argument("Each item length must be > 0");
        }
        if (v > L) {
            // Put oversized items in separate list
            oversized_items.emplace_back(v, (int)i);
        } else {
            items.emplace_back(v, (int)i);
        }
    }

    // Sort items decreasing by weight
    std::sort(items.begin(), items.end(), [](const std::pair<int,int> &a, const std::pair<int,int> &b){
        if (a.first != b.first) return a.first > b.first;
        return a.second < b.second;
    });

    std::vector<int> assignment(N, -1);
    std::vector<std::vector<int>> bin_to_items;
    std::vector<int> rem; // remaining capacities per bin

    if (mode == "ffd") {
        // First-Fit Decreasing: simply keep bins in order and scan for first fit
        for (const auto &it : items) {
            int w = it.first;
            int orig = it.second;
            int chosen = -1;
            for (size_t b = 0; b < rem.size(); ++b) {
                if (rem[b] >= w) { chosen = (int)b; break; }
            }
            if (chosen == -1) {
                // new bin
                chosen = (int)rem.size();
                rem.push_back(L);
                bin_to_items.emplace_back();
            }
            // assign
            rem[chosen] -= w;
            bin_to_items[chosen].push_back(orig);
            assignment[orig] = chosen;
        }
    } else if (mode == "bfd") {
        // Best-Fit Decreasing with segment tree optimization
        // space_to_bins[c] contains stack/vector of bin ids that currently have remaining capacity == c
        std::vector<std::vector<int>> space_to_bins(L+1);
        SegmentTree seg(L);

        for (const auto &it : items) {
            int w = it.first;
            int orig = it.second;
            int chosen_bin = -1;
            // find best fit capacity >= w
            int cap = seg.find_best_fit(w);
            if (cap == 0) {
                // no existing bin fits -> create new bin
                chosen_bin = (int)rem.size();
                rem.push_back(L);
                bin_to_items.emplace_back();
                // initially it has full capacity L: put into space_to_bins[L]
                if (L > 0) {
                    space_to_bins[L].push_back(chosen_bin);
                    seg.set_leaf(L, L);
                }
                // now we will pop it below
                cap = seg.find_best_fit(w);
                // cap should now be >= w (since newly added L >= w)
                if (cap == 0) {
                    throw std::runtime_error("failed to create initial bin");
                }
            }
            // pop a bin id from space_to_bins[cap]
            auto &vec = space_to_bins[cap];
            if (vec.empty()) {
                // inconsistent state
                seg.set_leaf(cap, 0);
                // requery
                cap = seg.find_best_fit(w);
                if (cap == 0) {
                    // create new bin as fallback
                    chosen_bin = (int)rem.size();
                    rem.push_back(L);
                    bin_to_items.emplace_back();
                    if (L>0) { space_to_bins[L].push_back(chosen_bin); seg.set_leaf(L, L); }
                    cap = seg.find_best_fit(w);
                }
            }
            if (chosen_bin == -1) {
                // pop
                auto &v2 = space_to_bins[cap];
                chosen_bin = v2.back(); v2.pop_back();
                if (v2.empty()) seg.set_leaf(cap, 0);
            }
            // assign item
            rem[chosen_bin] -= w;
            bin_to_items[chosen_bin].push_back(orig);
            assignment[orig] = chosen_bin;
            int new_r = rem[chosen_bin];
            if (new_r > 0) {
                space_to_bins[new_r].push_back(chosen_bin);
                seg.set_leaf(new_r, new_r);
            }
        }
    } else {
        throw std::invalid_argument("mode must be 'bfd' or 'ffd'");
    }

    // Handle oversized items - each gets its own bin
    for (const auto &it : oversized_items) {
        int orig = it.second;
        int bin_id = (int)rem.size();
        rem.push_back(0); // oversized item uses full capacity, so remaining is 0
        bin_to_items.emplace_back();
        bin_to_items[bin_id].push_back(orig);
        assignment[orig] = bin_id;
    }

    // Prepare return values
    py::array_t<int> assign_out(N);
    auto buf_out = assign_out.mutable_unchecked<1>();
    for (ssize_t i = 0; i < N; ++i) buf_out(i) = assignment[i];

    py::list py_bins;
    for (auto &b : bin_to_items) {
        py::list l;
        for (int idx : b) l.append(idx);
        py_bins.append(l);
    }
    py::list py_rem;
    for (int r : rem) py_rem.append(r);

    py::dict out;
    out["assignments"] = assign_out;
    out["bins"] = py_bins;
    out["rem"] = py_rem;
    return out;
}

// Pack documents and build new document index with padding
// Inputs:
// - document_index: 1D integer numpy array of document indices
// - sequence_lengths: 1D integer numpy array of sequence lengths for each document
// - seq_length: bin capacity (max sequence length)
// - mode: "bfd" or "ffd"
// - pad_marker: integer value to use for padding
// Returns:
// - new_document_index: 1D integer numpy array with packed documents and padding
py::array_t<int> pack_documents(py::array_t<int, py::array::c_style | py::array::forcecast> document_index,
                                py::array_t<int, py::array::c_style | py::array::forcecast> sequence_lengths,
                                int seq_length,
                                std::string mode = "ffd",
                                int pad_marker = -1) {
    auto doc_buf = document_index.unchecked<1>();
    auto seq_buf = sequence_lengths.unchecked<1>();
    ssize_t N = doc_buf.shape(0);

    // Extract sequence lengths for the given document indices
    std::vector<int> lengths(N);
    for (ssize_t i = 0; i < N; ++i) {
        int doc_idx = doc_buf(i);
        if (doc_idx < 0 || doc_idx >= seq_buf.shape(0)) {
            throw std::invalid_argument("Document index out of range");
        }
        lengths[i] = seq_buf(doc_idx);
    }

    // Create numpy array from lengths vector
    py::array_t<int> lengths_array(N);
    auto lengths_buf = lengths_array.mutable_unchecked<1>();
    for (ssize_t i = 0; i < N; ++i) {
        lengths_buf(i) = lengths[i];
    }

    // Pack the items
    py::dict pack_result = pack_items(lengths_array, seq_length, mode);

    // Extract results
    auto assignments = pack_result["assignments"].cast<py::array_t<int>>();
    auto bins = pack_result["bins"].cast<py::list>();
    auto rem = pack_result["rem"].cast<py::list>();

    // Build new document index
    std::vector<int> new_document_index;

    for (size_t bin_idx = 0; bin_idx < bins.size(); ++bin_idx) {
        py::list bin_items = bins[bin_idx].cast<py::list>();
        int remaining_size = rem[bin_idx].cast<int>();

        // Add document indices for this bin
        for (size_t item_idx = 0; item_idx < bin_items.size(); ++item_idx) {
            int local_index = bin_items[item_idx].cast<int>();
            int original_doc_index = doc_buf(local_index);
            new_document_index.push_back(original_doc_index);
        }

        // Add padding if needed
        if (remaining_size > 0) {
            for (int i = 0; i < remaining_size; ++i) {
                new_document_index.push_back(pad_marker);
            }
        } else if (remaining_size < 0) {
            // This handles oversized items - pad to seq_length
            int padding_needed = seq_length + remaining_size; // remaining_size is negative
            for (int i = 0; i < padding_needed; ++i) {
                new_document_index.push_back(pad_marker);
            }
        }
    }

    // Convert to numpy array
    py::array_t<int> result(new_document_index.size());
    auto result_buf = result.mutable_unchecked<1>();
    for (size_t i = 0; i < new_document_index.size(); ++i) {
        result_buf(i) = new_document_index[i];
    }

    return result;
}

PYBIND11_MODULE(helpers_cpp, m)
{
  m.def("build_mapping", &build_mapping);
  m.def("build_blocks_mapping", &build_blocks_mapping);
  m.def("build_sample_idx_int32", &build_sample_idx<int32_t>);
  m.def("build_sample_idx_int64", &build_sample_idx<int64_t>);
  m.def("build_blending_indices", &build_blending_indices);
  m.def("build_exhaustive_blending_indices", &build_exhaustive_blending_indices);
  m.def("pack_items", &pack_items,
        py::arg("lengths"), py::arg("L"), py::arg("mode") = "bfd",
        "Pack items (lengths) into bins of capacity L. mode = 'bfd' or 'ffd'.\n\n"
        "Returns dict with keys: 'assignments' (numpy array of bin ids), 'bins' (list of lists of item indices), 'rem' (remaining capacities)");
  m.def("pack_documents", &pack_documents,
        py::arg("document_index"), py::arg("sequence_lengths"), py::arg("seq_length"),
        py::arg("mode") = "ffd", py::arg("pad_marker") = -1,
        "Pack documents and build new document index with padding.\n\n"
        "Args:\n"
        "  document_index: 1D array of document indices\n"
        "  sequence_lengths: 1D array of sequence lengths for each document\n"
        "  seq_length: bin capacity (max sequence length)\n"
        "  mode: 'bfd' or 'ffd' packing mode\n"
        "  pad_marker: integer value to use for padding\n\n"
        "Returns:\n"
        "  new_document_index: 1D array with packed documents and padding");
}
