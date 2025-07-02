#include <stdexcept>
#include <algorithm>
#include <MTensor/tensorImpl.hpp>
#include <MTensor/ops.hpp>

namespace mt {
namespace ops{

    int64_t Permute::count = 0;

    Permute::Permute(const std::vector<int64_t>& dims_permute  , bool inc_counter )
    {
        // Check that all elements are unique
        std::vector<int64_t> sorted_dims = dims_permute;
        std::sort(sorted_dims.begin(), sorted_dims.end());
        for (size_t i = 1; i < sorted_dims.size(); ++i) {
            if (sorted_dims[i] == sorted_dims[i - 1]) {
                throw std::invalid_argument("error: Permute() dims_permute contains duplicate elements");
            }
        }

        // Check that min > -1 and max < size()
        auto minmax = std::minmax_element(dims_permute.begin(), dims_permute.end());
        if (*minmax.first < 0 || *minmax.second >= static_cast<int64_t>(dims_permute.size())) {
            throw std::out_of_range("error: Permute() dims_permute elements out of valid range");
        }

        m_dims_permute = dims_permute;

        if(inc_counter){
            m_name = "Permute"+std::to_string(count);
            count++;
        }

    } 
 
    std::shared_ptr<TensorImpl> Permute::forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) {

        const auto& in_tensor = operands[0];

        if (m_dims_permute.size() != in_tensor->shape().size()) {
            throw std::invalid_argument("error: Permute() m_dims_permute size does not match input tensor dimensions");
        }


        const auto& in_shape = in_tensor->shape();
        const auto& in_stride = in_tensor->stride();
        std::vector<int64_t> out_stride(in_stride.size());
        std::vector<int64_t> out_shape(in_shape.size());

        for (size_t i = 0 ; i < m_dims_permute.size() ; i++){
            out_shape[i] = in_shape[m_dims_permute[i]];
            out_stride[i] = in_stride[m_dims_permute[i]];
        }

        std::shared_ptr<Operation> grad_fn = nullptr;
        bool requires_grad = false;
        bool is_contiguous = m_dims_permute == in_shape ? in_tensor->is_contiguous() :  false ;

        if ( in_tensor->requires_grad() ){
            requires_grad = true; 
            grad_fn = std::make_shared<Permute>(m_dims_permute , true);
            grad_fn->set_operands({in_tensor});
        }

        return std::make_shared<TensorImpl>(in_tensor->data_ptr(),in_tensor->data_offset(), out_shape , grad_fn , requires_grad, is_contiguous, out_stride);
    }

    void Permute::backward(const std::shared_ptr<TensorImpl>& diff_loss_out){

    }

}//ops
}//mt