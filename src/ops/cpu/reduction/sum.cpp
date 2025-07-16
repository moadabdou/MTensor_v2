#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <MTensor/tensorImpl.hpp>
#include <MTensor/ops.hpp>

namespace mt {
namespace ops{

    int64_t Sum::count = 0;

    Sum::Sum(const std::vector<int64_t>& dims, bool inc_counter)
    {
        if (*std::min_element(dims.begin(), dims.end()) < 0){
            throw std::invalid_argument("error : Sum() invalid dims are given");
        }

        m_dims = dims;
        max_allowed_dim = *std::max_element(dims.begin(), dims.end());

        if(inc_counter){
            m_name = "Sum"+std::to_string(count);
            count++;
        }
    } 
 
    std::shared_ptr<TensorImpl> Sum::forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) {
        

        const auto& in_tensor = operands[0];


        const auto& in_shape = in_tensor->shape();
        const auto& src_stride = in_tensor->stride();

        if (max_allowed_dim >= in_shape.size()){
            throw std::invalid_argument("error: Sum() dims are out of range of in_tensor dims");
        }

        std::vector<int64_t> out_shape = in_shape;

        for(auto&& dim: m_dims)
            out_shape[dim] = 1;

        if (out_shape == in_shape){
            return in_tensor;
        }

        const auto& dst_stride = row_major_stride(out_shape);

        dnnl::memory::desc src_md(in_shape, dnnl::memory::data_type::f32, src_stride);
        dnnl::memory::desc dst_md(out_shape, dnnl::memory::data_type::f32, dst_stride);

        dnnl::engine eng(dnnl::engine::kind::cpu, 0);

        dnnl::reduction::primitive_desc reduction_desc(
            eng,  
            dnnl::algorithm::reduction_sum, 
            src_md, 
            dst_md,
            0.0f,
            0.0f
        );

        const auto& data_storage = custom_reduction_op(
            in_tensor, 
            reduction_desc, 
            eng, 
            src_md, 
            dst_md
        );

        std::shared_ptr<Operation> grad_fn = nullptr;
        bool requires_grad = false;

        if ( in_tensor->requires_grad() ){
            requires_grad = true; 
            grad_fn = std::make_shared<Sum>(m_dims, true);
            grad_fn->set_operands({in_tensor});
        }

        return std::make_shared<TensorImpl>(data_storage , 0 , out_shape , grad_fn , requires_grad, true , dst_stride);
    }  

    void Sum::backward(const std::shared_ptr<TensorImpl>& diff_loss_out){
//if (diff_loss_out->requires_grad()) std::cout << m_name ; 
        dnnl::engine engine(dnnl::engine::kind::cpu, 0);
        dnnl::stream engine_stream(engine);

        auto& x = m_operands[0];

        if (! x->requires_grad()) return;

        Expand expand(x->shape());
        Contiguous contiguous;

        auto diff_src =  contiguous.forward({expand.forward({diff_loss_out})});

        
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