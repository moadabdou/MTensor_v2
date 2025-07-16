#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <MTensor/tensorImpl.hpp>
#include <MTensor/ops.hpp>

namespace mt {
namespace ops{

    int64_t Exp::count = 0;

    Exp::Exp(bool inc_counter )
    {
        if(inc_counter){
            m_name = "Exp"+std::to_string(count);
            count++;
        }
    } 
 
    std::shared_ptr<TensorImpl> Exp::forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) {
        

        const auto& in_tensor = operands[0]->is_contiguous() ?  operands[0] : Contiguous().forward({operands[0]});


        const auto& m_shape = in_tensor->shape();
        const auto& src_stride = in_tensor->stride();
        const auto& dst_stride = row_major_stride(m_shape);

        dnnl::memory::desc src_md(m_shape, dnnl::memory::data_type::f32, src_stride);
        dnnl::memory::desc dst_md(m_shape, dnnl::memory::data_type::f32, dst_stride);

        dnnl::engine eng(dnnl::engine::kind::cpu, 0);

        dnnl::eltwise_forward::primitive_desc exp_desc(
            eng, 
            dnnl::prop_kind::forward_inference, 
            dnnl::algorithm::eltwise_exp, 
            src_md, 
            dst_md 
        );

        const auto& data_storage = custom_eltwise_op(
            in_tensor, 
            exp_desc, 
            eng, 
            src_md, 
            dst_md
        );

        std::shared_ptr<Operation> grad_fn = nullptr;
        bool requires_grad = false;

        if ( in_tensor->requires_grad() ){
            requires_grad = true; 
            grad_fn = std::make_shared<Exp>(true);
            grad_fn->set_operands({in_tensor});
        }

        return std::make_shared<TensorImpl>(data_storage , 0 , m_shape , grad_fn , requires_grad, true , dst_stride);
    }  

    void Exp::backward(const std::shared_ptr<TensorImpl>& diff_loss_out){
//if (diff_loss_out->requires_grad()) std::cout << m_name ; 
        dnnl::engine engine(dnnl::engine::kind::cpu, 0);
        dnnl::stream engine_stream(engine);

        const auto& x = m_operands[0];
        if (! x->requires_grad()) return;

        auto src_md = dnnl::memory::desc(x->shape() , dnnl::memory::data_type::f32, x->stride());
        auto diff_md = src_md;

        // Forward primitive desc (hint)
        auto fwd_pd = dnnl::eltwise_forward::primitive_desc(
            engine,
            dnnl::prop_kind::forward_inference,
            dnnl::algorithm::eltwise_exp,
            src_md,
            src_md
        );

        // Backward primitive desc
        auto bwd_pd = dnnl::eltwise_backward::primitive_desc(
            engine,
            dnnl::algorithm::eltwise_exp,
            src_md,
            diff_md,
            src_md,
            fwd_pd
        );


        dnnl::memory src_mem(src_md, engine, x->data_ptr().get() + x->data_offset());
        dnnl::memory diff_dst_mem(diff_md, engine, diff_loss_out->data_ptr().get() + diff_loss_out->data_offset());
        dnnl::memory diff_src_mem;
        std::shared_ptr<float> data_storage; //in case if the x_grad does not exist

        if (x->get_grad()){
            diff_src_mem = dnnl::memory(diff_md, engine); //x_grad exists so we allocate new temporary memory
        }else{
            data_storage = std::shared_ptr<float>(new float[x->numel()], std::default_delete<float[]>());
            diff_src_mem = dnnl::memory(diff_md, engine, data_storage.get()); // x_grad does not exists so we make one and the result will directly routed to it 
        }

        auto eltwise_bwd = dnnl::eltwise_backward(bwd_pd);
        eltwise_bwd.execute(engine_stream, {
            {DNNL_ARG_SRC, src_mem},
            {DNNL_ARG_DIFF_DST, diff_dst_mem},
            {DNNL_ARG_DIFF_SRC, diff_src_mem}
        });

        engine_stream.wait();

        
        {
            if (x->get_grad()){
                accumulate(
                    diff_src_mem,
                    x->get_grad(),
                    engine,
                    engine_stream
                );
            }else {
                x->set_grad(std::make_shared<TensorImpl>(data_storage, 0 , diff_md.get_dims(), nullptr , false, true , diff_md.get_strides()));
            }
        }
    }

}//ops
}//mt