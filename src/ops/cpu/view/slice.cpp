#include <stdexcept>
#include <iostream>
#include <MTensor/tensorImpl.hpp>
#include <MTensor/ops.hpp>

namespace mt {
namespace ops{

    int64_t Slice::count = 0;

    Slice::Slice(const sliceList& slice_list  , bool inc_counter )
    {
        
        if (slice_list.empty()){
            throw std::invalid_argument("error: Slice() slice_list cannot be empty.");
        }

        m_slice_list = slice_list;

        if(inc_counter){
            m_name = "Slice"+std::to_string(count);
            count++;
        }

    } 
 
    std::shared_ptr<TensorImpl> Slice::forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) {

        const auto& in_tensor = operands[0];
        const auto& in_shape = in_tensor->shape();

        if (m_slice_list.size() > in_shape.size()) {
            throw std::invalid_argument("error: Slice() slice_list size cannot be greater than tensor dimensions.");
        }

        // Transform m_slice_list to absolute indices and Fill missing slices with full range
        sliceList abs_slice_list(in_shape.size());

        for (int64_t i = 0; i < in_shape.size(); ++i) {
            int64_t dim_size = in_shape[i];
            int64_t start = 0;
            int64_t end = dim_size;

            if (i < m_slice_list.size()) {
                start = m_slice_list[i].first;
                end = m_slice_list[i].second;

                if (end == EOD) 
                    end = dim_size;
                if (start == EOD) 
                    start = dim_size;

                // Handle negative indices
                if (start < 0) start += dim_size ;
                if (end < 0) end += dim_size ;

                // Clamp to bounds
                if (start < 0 || start > dim_size)
                    throw std::out_of_range("error: Slice() start index out of range for dimension " + std::to_string(i));
                if (end < start || end > dim_size)
                    throw std::out_of_range("error: Slice() end index out of range for dimension " + std::to_string(i));
            }

            abs_slice_list[i] = {start, end};
        }

        std::vector<int64_t> out_stride = in_tensor->stride();
        std::vector<int64_t> out_shape(in_shape.size());
        int64_t out_data_offset = in_tensor->data_offset();

        for (int64_t i = 0; i < abs_slice_list.size(); ++i) {
            int64_t start = abs_slice_list[i].first;
            int64_t end = abs_slice_list[i].second;
            out_shape[i] = end - start;
            out_data_offset += start * out_stride[i];
        }

        std::shared_ptr<Operation> grad_fn = nullptr;
        bool requires_grad = false;

        if ( in_tensor->requires_grad() ){
            requires_grad = true; 
            grad_fn = std::make_shared<Slice>(m_slice_list , true);
            grad_fn->set_operands({in_tensor});
        }

        return std::make_shared<TensorImpl>(in_tensor->data_ptr(),out_data_offset, out_shape , grad_fn , requires_grad,false, out_stride);
    }

    void Slice::backward(const std::shared_ptr<TensorImpl>& diff_out){
          dnnl::engine engine(dnnl::engine::kind::cpu, 0);
        dnnl::stream engine_stream(engine);

        auto& x = m_operands[0];
        if (! x->requires_grad()) return;

        
        {
        if (!x->get_grad()){
            x->set_grad(TensorImpl::zeros(x->shape()));
        }
        }

        Slice slice(m_slice_list);

        auto sliced_diff_src =  slice.forward({x->get_grad()});

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