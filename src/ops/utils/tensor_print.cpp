#include <iostream>
#include <iomanip>
#include <MTensor/tensorImpl.hpp>
#include <MTensor/utils/tensor_print.hpp>

namespace mt {
namespace utils {
    void print_tensor(const float* data, const int64_t data_offset,const std::vector<int64_t>& shape, const std::vector<int64_t>& stride,
        int64_t numel, int level , int64_t offset) {
        if (shape.empty()) {
            float value = data[data_offset + offset];
            if (std::abs(value) >= 9.0f || (value != 0.0f && std::abs(value) < 0.01f)) {
                std::cout << (value >= 0 ? " " : "") << std::scientific << std::setprecision(3) << value;
            } else {
                std::cout << (value >= 0 ? " " : "") << std::fixed << std::setprecision(7) << value ;
            }
            return;
        }
        int64_t dim = shape[0];
        std::vector<int64_t> sub_shape(shape.begin() + 1, shape.end());
        std::vector<int64_t> sub_stride(stride.begin() + 1, stride.end());

        std::cout << std::string(level * 2, ' ') << "[";

        for (int64_t i = 0; i < dim; ++i) {
            if (i > 0) std::cout << ", ";
            if (!sub_shape.empty()) std::cout << "\n";

            int64_t sub_offset = offset + i * stride[0];
            print_tensor(data, data_offset, sub_shape, sub_stride, numel, level + 1, sub_offset);
        }

        if (!sub_shape.empty()) std::cout << "\n" << std::string(level * 2, ' ');
        std::cout << "]";
    }

}//utils
}//mt