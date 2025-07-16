#include <stdexcept>
#include <MTensor/tensorImpl.hpp>
#include <MTensor/ops.hpp>

namespace mt {
namespace ops{

    int64_t Unsqueeze::count = 0;

    Unsqueeze::Unsqueeze(const int64_t& dim , bool inc_counter )
    {
        if (dim < 0 ){
            throw std::invalid_argument(
                "error: Unsqueeze() invalide dim was passed"
            );
        }
        m_dim = dim;
        if(inc_counter){
            m_name = "Unsqueeze"+std::to_string(count);
            count++;
        }
    } 
 
    std::shared_ptr<TensorImpl> Unsqueeze::forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) {

        const auto& in_tensor = operands[0];

        // Check that m_dim is within the valid range
        auto ndim = static_cast<int64_t>(in_tensor->shape().size());
        if (m_dim > ndim) {
            throw std::out_of_range(
                "error: Unsqueeze() dim is out of range of in_tensor"
            );
        }

        // Copy the input shape to out_shape
        auto out_shape = in_tensor->shape();
        auto out_stride = in_tensor->stride();

        // Insert dimension of size 1 at index m_dim in out_shape
        out_shape.insert(out_shape.begin() + m_dim, 1);
        // For stride, insert the same value as the previous dimension if possible, else 1
        if (m_dim < static_cast<int64_t>(out_stride.size())) {
            out_stride.insert(out_stride.begin() + m_dim, out_stride[m_dim]);
        } else {
            out_stride.push_back(1);
        }

        std::shared_ptr<Operation> grad_fn = nullptr;
        bool requires_grad = false;

        if ( in_tensor->requires_grad() ){
            requires_grad = true; 
            grad_fn = std::make_shared<Unsqueeze>(m_dim , true);
            grad_fn->set_operands({in_tensor});
        }

        return std::make_shared<TensorImpl>(in_tensor->data_ptr(), in_tensor->data_offset() , out_shape , grad_fn , requires_grad, in_tensor->is_contiguous() ,out_stride);
    }

    void Unsqueeze::backward(const std::shared_ptr<TensorImpl>& diff_loss_out){
//if (diff_loss_out->requires_grad()) std::cout << m_name ; 
        dnnl::engine engine(dnnl::engine::kind::cpu, 0);
        dnnl::stream engine_stream(engine);
        Squeeze squeeze(m_dim);
        Contiguous contiguous;

        auto& x = m_operands[0];
        if (! x->requires_grad()) return;

        auto diff_src =  contiguous.forward({squeeze.forward({diff_loss_out})});

        
        {
        if (x->get_grad()){

            auto diff_src_md = dnnl::memory::desc(diff_src->shape() , dnnl::memory::data_type::f32, diff_src->stride());
            dnnl::memory diff_src_mem(diff_src_md, engine, diff_src->data_ptr().get() + diff_src->data_offset());

            accumulate(
                diff_src_mem,
                x->get_grad(),
                engine,
                engine_stream
            );

        }else {
            x->set_grad(diff_src);
        }
        }
    }

}//ops
}//mt