#include <iostream>
#include <MTensor/tensorImpl.hpp>
#include <MTensor/utils/tensor_print.hpp>

namespace mt {
namespace utils {
    void print_tensor(const float* data, const int64_t data_offset,const std::vector<int64_t>& shape, const std::vector<int64_t>& stride,
        int64_t numel, int level , int64_t offset) {
        if (shape.empty()) {
            std::cout << data[data_offset + offset];
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


std::ostream& operator<<(std::ostream& os, const std::shared_ptr<mt::TensorImpl>& tensor_impl) {
    if (!tensor_impl) {
        os << "TensorImpl(nullptr)";
        return os;
    }
    const auto& shape = tensor_impl->shape();
    const auto& stride = tensor_impl->stride();
    const float* data = tensor_impl->data_ptr().get();
    const int64_t data_offset = tensor_impl->data_offset();
    int64_t numel = tensor_impl->numel();
    os << "Tensor(";
    mt::utils::print_tensor(data,data_offset, shape, stride, numel);
    
    if (tensor_impl->is_leaf()){
        if (tensor_impl->requires_grad()){
            std::cout << ", requires_grad = True";
        }
    }else {
        std::cout << ", grad_fn = "<< tensor_impl->grad_fn()->name();
    }
    
    std::cout << ")"<< std::endl;
    return os;
}