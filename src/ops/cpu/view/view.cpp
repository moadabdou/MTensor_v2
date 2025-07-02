#include <numeric>
#include <stdexcept>
#include <MTensor/tensorImpl.hpp>
#include <MTensor/ops.hpp>

namespace mt {
namespace ops{

    int64_t View::count = 0;

    View::View(const std::vector<int64_t>& shape, bool inc_counter )
    {
        if (! TensorImpl::is_valid_shape(shape)){
            throw std::invalid_argument("error : View() invalid shape is given");
        }
        if(inc_counter){
            m_name = "View"+std::to_string(count);
            count++;
        }
        m_shape = shape;
        m_out_numel = std::accumulate(m_shape.begin(), m_shape.end(),1ull, std::multiplies<int64_t>());
    } 
 
    std::shared_ptr<TensorImpl> View::forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) {

        const auto& in_tensor = operands[0];

        if (!in_tensor->is_contiguous()) {
            throw std::runtime_error("view() requires contiguous tensor");
        }

        if (m_out_numel != in_tensor->numel()){
            throw std::invalid_argument(
                std::string("error: view() in_tensor numel and view_op_shape product are not equal view's shape_product:") 
                + std::to_string(m_out_numel) 
                + " & in_tensor's numel:"
                + std::to_string(in_tensor->numel()) 
            );
        }

        std::shared_ptr<Operation> grad_fn = nullptr;
        bool requires_grad = false;

        if ( in_tensor->requires_grad() ){
            requires_grad = true; 
            grad_fn = std::make_shared<View>(m_shape, true);
            grad_fn->set_operands({in_tensor});
        }

        return std::make_shared<TensorImpl>(in_tensor->data_ptr(),in_tensor->data_offset(), m_shape, grad_fn , requires_grad, true);

    }  

    void View::backward(const std::shared_ptr<TensorImpl>& diff_loss_out){

    }

}//ops
}//mt


