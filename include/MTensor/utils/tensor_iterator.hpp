#include <vector>
#include <stdexcept>
#include <omp.h>

namespace mt {
namespace utils{

class TensorIterator {
public:
    TensorIterator(std::vector<long long> shape, std::vector<long long> strides);

    long long get_total_elements() const { return total_elements_; }

    long long get_flat_index_from_coords(const std::vector<long long>& coords) const;

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
    std::vector<long long> get_coords_from_iteration_num(long long iteration_num) const ;

    void increment_coords_by_one(std::vector<long long>& coords) const;

    std::vector<long long> shape_;
    std::vector<long long> strides_;
    long long total_elements_;
};   

}//utils

}// mt 
