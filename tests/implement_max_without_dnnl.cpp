#include <iostream>
#include <vector>
#include <stdexcept>
#include <numeric>
#include <functional>
#include <limits>
#include <omp.h>
#include <algorithm> // For std::min
#include <immintrin.h> // For AVX512
#include <memory> // For std::shared_ptr

// The TensorIterator class from the previous step is included here for completeness.
#pragma region TensorIterator
/**
 * @class TensorIterator
 * @brief A high-performance, parallel iterator for traversing tensor elements.
 */
class TensorIterator {
public:
    TensorIterator(std::vector<long long> shape, std::vector<long long> strides)
        : shape_(std::move(shape)), strides_(std::move(strides)) {
        if (shape_.size() != strides_.size()) {
            throw std::invalid_argument("Shape and strides must have the same number of dimensions.");
        }
        total_elements_ = 1;
        for (long long dim_size : shape_) {
            if (dim_size < 0) throw std::invalid_argument("Dimension size cannot be negative.");
            total_elements_ *= dim_size;
        }
    }

    long long get_total_elements() const { return total_elements_; }

    long long get_flat_index_from_coords(const std::vector<long long>& coords) const {
        long long flat_index = 0;
        for (size_t i = 0; i < coords.size(); ++i) {
            flat_index += coords[i] * strides_[i];
        }
        return flat_index;
    }

    template<typename F>
    void parallel_for_each(const F& func) const {
        #pragma omp parallel
        {
            int num_threads = omp_get_num_threads();
            int thread_id = omp_get_thread_num();
            long long items_per_thread = total_elements_ / num_threads;
            long long start_k = thread_id * items_per_thread;
            long long end_k = (thread_id == num_threads - 1) ? total_elements_ : start_k + items_per_thread;

            if (start_k < end_k) {
                std::vector<long long> coords = get_coords_from_iteration_num(start_k);
                for (long long k = start_k; k < end_k; ++k) {
                    func(coords, k); // Pass coordinates and logical index
                    if (k < end_k - 1) {
                        increment_coords_by_one(coords);
                    }
                }
            }
        }
    }

private:
    std::vector<long long> get_coords_from_iteration_num(long long iteration_num) const {
        std::vector<long long> coords(shape_.size(), 0);
        long long current_index = iteration_num;
        for (int i = shape_.size() - 1; i >= 0; --i) {
            if (shape_[i] > 0) {
                coords[i] = current_index % shape_[i];
                current_index /= shape_[i];
            }
        }
        return coords;
    }

    void increment_coords_by_one(std::vector<long long>& coords) const {
        for (int i = shape_.size() - 1; i >= 0; --i) {
            coords[i]++;
            if (coords[i] < shape_[i]) return;
            coords[i] = 0;
        }
    }

    std::vector<long long> shape_;
    std::vector<long long> strides_;
    long long total_elements_;
};
#pragma endregion

/**
 * @brief Finds the maximum value and its index from an AVX512 register.
 * @param max_vals_vec An __m512 register holding 16 float values.
 * @param max_indices_vec An __m512i register holding 16 int32 indices.
 * @return A pair containing the maximum float value and its index.
 */
std::pair<float, int> horizontal_max_with_index_512(__m512 max_vals_vec, __m512i max_indices_vec) {
    // This is a pattern for horizontal reduction on AVX512.
    // It finds the max value and keeps its corresponding index paired.
    
    // Step 1: Reduce 16 -> 8
    __m256 half_vals = _mm512_extractf32x8_ps(max_vals_vec, 1);
    __m256i half_indices = _mm512_extracti32x8_epi32(max_indices_vec, 1);
    __m256 max_vals_vec_256 = _mm512_castps512_ps256(max_vals_vec);
    __m256i max_indices_vec_256 = _mm512_castsi512_si256(max_indices_vec);
    __m256 cmp_mask8 = _mm256_cmp_ps(max_vals_vec_256, half_vals, _CMP_LT_OQ);
    __m256 max_vals_256 = _mm256_blendv_ps(max_vals_vec_256, half_vals, cmp_mask8);
    __m256i cmp_mask8i = _mm256_castps_si256(cmp_mask8);
    __m256i max_indices_256 = _mm256_blendv_epi8(max_indices_vec_256, half_indices, cmp_mask8i);

    // Step 2: Reduce 8 -> 4
    __m128 half_vals_128 = _mm256_extractf128_ps(max_vals_256, 1);
    __m128i half_indices_128 = _mm256_extracti128_si256(max_indices_256, 1);
    __mmask8 mask4 = _mm_cmp_ps_mask(_mm256_castps256_ps128(max_vals_256), half_vals_128, _CMP_LT_OQ);
    __m128 max_vals_128 = _mm_mask_blend_ps(mask4, _mm256_castps256_ps128(max_vals_256), half_vals_128);
    __m128i max_indices_128 = _mm_mask_blend_epi32(mask4, _mm256_castsi256_si128(max_indices_256), half_indices_128);

    // Step 3: Reduce 4 -> 2
    half_vals_128 = _mm_movehl_ps(max_vals_128, max_vals_128);
    half_indices_128 = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(max_indices_128), _mm_castsi128_ps(max_indices_128)));
    __mmask8 mask2 = _mm_cmp_ps_mask(max_vals_128, half_vals_128, _CMP_LT_OQ);
    max_vals_128 = _mm_mask_blend_ps(mask2, max_vals_128, half_vals_128);
    max_indices_128 = _mm_mask_blend_epi32(mask2, max_indices_128, half_indices_128);

    // Step 4: Reduce 2 -> 1
    half_vals_128 = _mm_shuffle_ps(max_vals_128, max_vals_128, _MM_SHUFFLE(1, 1, 1, 1));
    half_indices_128 = _mm_shuffle_epi32(max_indices_128, _MM_SHUFFLE(1, 1, 1, 1));
    __mmask8 mask1 = _mm_cmp_ss_mask(max_vals_128, half_vals_128, _CMP_LT_OQ);
    max_vals_128 = _mm_mask_blend_ps(mask1, max_vals_128, half_vals_128);
    max_indices_128 = _mm_mask_blend_epi32(mask1, max_indices_128, half_indices_128);
    
    return { _mm_cvtss_f32(max_vals_128), _mm_cvtsi128_si32(max_indices_128) };
}


/**
 * @brief Reduces a tensor on its last dimension using AVX512, finding the max value and its index.
 *
 * @param data_ptr Pointer to the start of the (contiguous) input tensor data.
 * @param shape The shape of the input tensor.
 * @param strides The strides of the input tensor.
 * @param max_indices_vec A reference to a vector that will be filled with the indices.
 * @return A shared_ptr to a newly allocated tensor containing the maximum values.
 */
std::shared_ptr<float[]> reduce_max_last_dim_avx512(
    const float* data_ptr,
    const std::vector<long long>& shape,
    const std::vector<long long>& strides,
    std::vector<long long>& max_indices_vec
) {
    if (shape.empty()) {
        throw std::runtime_error("Input tensor cannot be a scalar.");
    }

    // --- 1. Setup Output and Iterator ---
    const long long last_dim_size = shape.back();
    std::vector<long long> outer_shape(shape.begin(), shape.end() - 1);
    std::vector<long long> outer_strides(strides.begin(), strides.end() - 1);

    long long output_size = 1;
    for (long long dim : outer_shape) output_size *= dim;

    // Use a smart pointer for automatic memory management.
    auto output_ptr = std::shared_ptr<float[]>(new float[output_size], std::default_delete<float[]>());
    max_indices_vec.resize(output_size);

    TensorIterator it(outer_shape, outer_strides);

    // --- 2. Define the Parallel Kernel ---
    auto kernel = [&](const std::vector<long long>& outer_coords, long long logical_idx) {
        const long long slice_start_offset = it.get_flat_index_from_coords(outer_coords);
        const float* slice_ptr = data_ptr + slice_start_offset;

        float max_val = -std::numeric_limits<float>::infinity();
        long long max_idx = -1;

        // --- 3. AVX512 Vectorized Reduction ---
        const int vec_width = 16; // 16 floats in __m512
        long long vec_end = (last_dim_size / vec_width) * vec_width;

        if (vec_end > 0) {
            __m512 max_vals_vec = _mm512_loadu_ps(slice_ptr);
            __m512i max_indices_vec_i = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

            for (long long i = vec_width; i < vec_end; i += vec_width) {
                __m512 next_vals_vec = _mm512_loadu_ps(slice_ptr + i);
                __m512i next_indices_vec_i = _mm512_add_epi32(max_indices_vec_i, _mm512_set1_epi32(vec_width));
                
                // Create a mask where next > current max
                __mmask16 mask = _mm512_cmp_ps_mask(next_vals_vec, max_vals_vec, _CMP_GT_OQ);

                // Use the mask to blend values from either the current max or the next vector
                max_vals_vec = _mm512_mask_blend_ps(mask, max_vals_vec, next_vals_vec);
                max_indices_vec_i = _mm512_mask_blend_epi32(mask, max_indices_vec_i, next_indices_vec_i);
            }
            
            auto result = horizontal_max_with_index_512(max_vals_vec, max_indices_vec_i);
            max_val = result.first;
            max_idx = result.second;
        }

        // --- 4. Scalar Reduction for Remainder ---
        // A simple scalar loop is robust and clear. The performance gain from
        // vectorizing/unrolling a tiny remainder (max 15 elements) is negligible.
        for (long long i = vec_end; i < last_dim_size; ++i) {
            if (slice_ptr[i] > max_val) {
                max_val = slice_ptr[i];
                max_idx = i;
            }
        }

        // --- 5. Store Results ---
        output_ptr[logical_idx] = max_val;
        max_indices_vec[logical_idx] = max_idx;
    };

    // --- 6. Execute ---
    it.parallel_for_each(kernel);

    return output_ptr;
}


int main() {
    // --- Setup Input Tensor ---
    // Shape (2, 3, 20) to have a non-multiple of 16 last dim
    const std::vector<long long> shape = {2, 3, 20};
    const std::vector<long long> strides = {3 * 20, 20, 1}; // C-contiguous
    long long total_size = shape[0] * shape[1] * shape[2];
    
    std::vector<float> data(total_size);
    // Fill with some data, making sure the max is unique in each slice
    for(long long i = 0; i < shape[0]; ++i) {
        for(long long j = 0; j < shape[1]; ++j) {
            for(long long k = 0; k < shape[2]; ++k) {
                long long offset = i * strides[0] + j * strides[1] + k * strides[2];
                if (k == (i + j + 3) % shape[2]) {
                    data[offset] = 100.0f + i + j;
                } else {
                    data[offset] = (float)(k);
                }
            }
        }
    }

    std::cout << "--- Input Tensor (first slice) ---" << std::endl;
    for(long long k=0; k<shape[2]; ++k) std::cout << data[k] << " ";
    std::cout << std::endl << std::endl;

    // --- Execute Reduction ---
    std::vector<long long> max_indices;
    // The returned pointer is a shared_ptr, no need to delete it manually.
    std::shared_ptr<float[]> output_data = reduce_max_last_dim_avx512(data.data(), shape, strides, max_indices);

    // --- Print Results ---
    std::cout << "--- Reduction Results ---" << std::endl;
    long long output_idx = 0;
    for(long long i = 0; i < shape[0]; ++i) {
        for(long long j = 0; j < shape[1]; ++j) {
            std::cout << "Max of slice (" << i << "," << j << "): "
                      << output_data[output_idx]
                      << " found at index " << max_indices[output_idx] << std::endl;
            output_idx++;
        }
    }

    // No need to call delete[] output_data; shared_ptr handles it.
    return 0;
}
