#include <numeric>
#include <stdexcept>
#include <MTensor/tensorImpl.hpp>
#include <MTensor/ops.hpp>

namespace mt {
namespace ops{

    int64_t Expand::count = 0;

    Expand::Expand(const std::vector<int64_t>& shape, bool inc_counter )
    {
        if (! TensorImpl::is_valid_shape(shape)){
            throw std::invalid_argument("error : Expand() invalid shape is given");
        }
        if(inc_counter){
            m_name = "Expand"+std::to_string(count);
            count++;
        }
        m_shape = shape;
    } 
 
    std::shared_ptr<TensorImpl> Expand::forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) {

        const auto& in_tensor = operands[0];

        if (m_shape.size() < in_tensor->shape().size()){
            throw std::length_error(
                "error: Expand(), in_tensor shape is bigger than the shape of expand"
            );
        }

        // Check if m_shape is a valid expansion of in_tensor->shape()
        const auto& in_shape = in_tensor->shape();
        const auto& in_stride = in_tensor->stride();

        int64_t in_dim = in_shape.size();
        int64_t out_dim = m_shape.size();
        int64_t lower_bound = out_dim - in_dim;

        std::vector<int64_t> out_stride(out_dim, 0); 

        // Align dimensions from the right
        for (int64_t i = out_dim - 1; i > lower_bound  - 1 ; --i) {
            int64_t in_size  = in_shape[i - lower_bound ];
            int64_t out_size = m_shape[i];

            if (in_size != out_size && in_size != 1) {
                throw std::invalid_argument(
                    "error: Expand(), m_shape is not a valid expansion of input shape"
                );
            }

            if (in_size == out_size){
                out_stride[i] = in_stride[i - lower_bound ];
            }else if (in_size == 1){
                out_stride[i] = 0;
            }

        }

        

        std::shared_ptr<Operation> grad_fn = nullptr;
        bool requires_grad = false;

        if ( in_tensor->requires_grad() ){
            requires_grad = true; 
            grad_fn = std::make_shared<Expand>(m_shape, true);
            grad_fn->set_operands({in_tensor});
        }

        return std::make_shared<TensorImpl>(in_tensor->data_ptr(), in_tensor->data_offset() , m_shape, grad_fn , requires_grad,false , out_stride);

    }

    void Expand::backward(const std::shared_ptr<TensorImpl>& diff_loss_out){

    }

}//ops
}//mt


