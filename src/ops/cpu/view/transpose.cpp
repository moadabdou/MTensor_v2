#include <stdexcept>
#include <MTensor/tensorImpl.hpp>
#include <MTensor/ops.hpp>

namespace mt {
namespace ops{

    int64_t Transpose::count = 0;

    Transpose::Transpose(const int64_t dim0,const int64_t dim1 , bool inc_counter )
    {
        if (dim0 < 0 || dim1 < 0){
            throw std::invalid_argument(
                "error: Transpose() invalide dim0 or dim1 was passed"
            );
        }
        m_dim0 = dim0;
        m_dim1 = dim1;
        if(inc_counter){
            m_name = "Transpose"+std::to_string(count);
            count++;
        }
    } 
 
    std::shared_ptr<TensorImpl> Transpose::forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) {

        const auto& in_tensor = operands[0];

        // Check that m_dim0 and m_dim1 are within the valid range
        auto ndim = static_cast<int64_t>(in_tensor->shape().size());
        if (m_dim0 >= ndim || m_dim1 >= ndim) {
            throw std::out_of_range(
                "error: Transpose() dim0 or dim1 is out of range"
            );
        }

        std::vector<int64_t> out_stride = in_tensor->stride();
        int64_t tmp_dim = out_stride[m_dim0];
        out_stride[m_dim0] = out_stride[m_dim1];
        out_stride[m_dim1] = tmp_dim;

        std::vector<int64_t> out_shape = in_tensor->shape();
        int64_t tmp_shape = out_shape[m_dim0];
        out_shape[m_dim0] = out_shape[m_dim1];
        out_shape[m_dim1] = tmp_shape;

        std::shared_ptr<Operation> grad_fn = nullptr;
        bool requires_grad = false;

        if ( in_tensor->requires_grad() ){
            requires_grad = true; 
            grad_fn = std::make_shared<Transpose>(m_dim0, m_dim1, true);
            grad_fn->set_operands({in_tensor});
        }

        return std::make_shared<TensorImpl>(in_tensor->data_ptr(),in_tensor->data_offset(), out_shape , grad_fn , requires_grad, false ,out_stride);
    }  

    void Transpose::backward(const std::shared_ptr<TensorImpl>& diff_loss_out){

    }

}//ops
}//mt