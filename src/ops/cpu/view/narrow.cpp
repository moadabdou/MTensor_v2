#include <stdexcept>
#include <MTensor/tensorImpl.hpp>
#include <MTensor/ops.hpp>

namespace mt {
namespace ops{

    int64_t Narrow::count = 0;

    Narrow::Narrow(const int64_t& dim, const int64_t& start, const int64_t& length , bool inc_counter )
    {
        if (dim < 0 ){
            throw std::invalid_argument(
                "error: Narrow() invalide dim was passed"
            );
        }

        if ( start < 0 || length < 1 ){
            throw std::invalid_argument(
                "error: Narrow() invalide start or length was passed, allowed range start -> [0,..[ and length -> [1,..[ "
            );
        }

        m_dim = dim;
        m_start = start;
        m_length = length;

        if(inc_counter){
            m_name = "Narrow"+std::to_string(count);
            count++;
        }
    } 
 
    std::shared_ptr<TensorImpl> Narrow::forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) {

        const auto& in_tensor = operands[0];

        // Check that m_dim0 and m_dim1 are within the valid range
        auto ndim = static_cast<int64_t>(in_tensor->shape().size());


        if (m_dim >= ndim) {
            throw std::invalid_argument(
                "error: Narrow() dim is out of range of in_tensor"
            );
        }

        // Copy the input shape to out_shape
        auto out_shape = in_tensor->shape();
        const auto& out_stride = in_tensor->stride();

        //check if the slice is withen the allowed range
        if (m_start + m_length > out_shape[m_dim]) {
            throw std::out_of_range(
                "error: Narrow() slice range is out of range the allowed range, slice_range[start, start+length] in [0, dim]"
            );
        }

        out_shape[m_dim] = m_length;
        int64_t out_data_offset = in_tensor->data_offset() + m_start*out_stride[m_dim];

        std::shared_ptr<Operation> grad_fn = nullptr;
        bool requires_grad = false;

        if ( in_tensor->requires_grad() ){
            requires_grad = true; 
            grad_fn = std::make_shared<Narrow>(m_dim, m_start, m_length , true);
            grad_fn->set_operands({in_tensor});
        }

        return std::make_shared<TensorImpl>(in_tensor->data_ptr(),out_data_offset, out_shape , grad_fn , requires_grad,false, out_stride);
    }

    void Narrow::backward(const std::shared_ptr<TensorImpl>& diff_out){
        dnnl::engine engine(dnnl::engine::kind::cpu, 0);
        dnnl::stream engine_stream(engine);

        auto& x = m_operands[0];
        if (! x->requires_grad()) return;

        
        {
        if (!x->get_grad()){
            x->set_grad(TensorImpl::zeros(x->shape()));
        }
        }

        Narrow narrow(m_dim, m_start, m_length);

        auto sliced_diff_src =  narrow.forward({x->get_grad()});

        auto diff_out_md = dnnl::memory::desc(diff_out->shape() , dnnl::memory::data_type::f32, diff_out->stride());
        dnnl::memory diff_out_mem(diff_out_md, engine, diff_out->data_ptr().get() + diff_out->data_offset());

        
        {
            accumulate(
                diff_out_mem,
                sliced_diff_src,
                engine,
                engine_stream
            );
        }
    }

}//ops
}//mt