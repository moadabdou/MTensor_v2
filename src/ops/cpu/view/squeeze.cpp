#include <stdexcept>
#include <MTensor/tensorImpl.hpp>
#include <MTensor/ops.hpp>

namespace mt {
namespace ops{

    int64_t Squeeze::count = 0;

    Squeeze::Squeeze(const int64_t& dim , bool inc_counter )
    {
        if (dim < 0 ){
            throw std::invalid_argument(
                "error: Squeeze() invalide dim was passed"
            );
        }
        m_dim = dim;
        if(inc_counter){
            m_name = "Squeeze"+std::to_string(count);
            count++;
        }
    } 
 
    std::shared_ptr<TensorImpl> Squeeze::forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) {

        const auto& in_tensor = operands[0];

        // Check that m_dim is within the valid range
        auto ndim = static_cast<int64_t>(in_tensor->shape().size());
        if (m_dim >= ndim) {
            throw std::out_of_range(
                "error: Squeeze() dim is out of range of in_tensor"
            );
        }

        // Copy the input shape to out_shape
        auto out_shape = in_tensor->shape();
        auto out_stride = in_tensor->stride();

        // Check if the dimension to squeeze is 1
        if (out_shape[m_dim] == 1) {
            // Remove the dimension from out_shape and out_stride
            out_shape.erase(out_shape.begin() + m_dim);
            out_stride.erase(out_stride.begin() + m_dim);
        }

        std::shared_ptr<Operation> grad_fn = nullptr;
        bool requires_grad = false;

        if ( in_tensor->requires_grad() ){
            requires_grad = true; 
            grad_fn = std::make_shared<Squeeze>(m_dim , true);
            grad_fn->set_operands({in_tensor});
        }

        return std::make_shared<TensorImpl>(in_tensor->data_ptr(), in_tensor->data_offset(), out_shape , grad_fn , requires_grad, in_tensor->is_contiguous(), out_stride);
    }  

    void Squeeze::backward(const std::shared_ptr<TensorImpl>& diff_loss_out){

    }

}//ops
}//mt