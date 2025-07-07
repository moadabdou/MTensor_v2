#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <MTensor/tensorImpl.hpp>
#include <MTensor/ops.hpp>

namespace mt {
namespace ops{

    int64_t Conv3d::count = 0;

    Conv3d::Conv3d(
        const std::vector<int64_t>& strides,
        const std::vector<int64_t>& padding_l,
        const std::vector<int64_t>& padding_r,
        bool inc_counter)
    {

        if (strides.size() != 3 || strides[0] < 0 || strides[1] < 0 || strides[2] < 0){
            throw std::invalid_argument("error : Conv3d() invalid strides were given");
        }
        if (padding_l.size() != 3 || padding_l[0] < 0 || padding_l[1] < 0 || padding_l[2] < 0){
            throw std::invalid_argument("error : Conv3d() invalid padding_l was given");
        }
        if (padding_r.size() != 3 || padding_r[0] < 0 || padding_r[1] < 0 || padding_r[2] < 0 ){
            throw std::invalid_argument("error : Conv3d() invalid padding_r was given");
        }

        m_strides = strides;
        m_padding_l = padding_l;
        m_padding_r = padding_r;

        if(inc_counter){
            m_name = "Conv3d"+std::to_string(count);
            count++;
        }
    } 
 
    std::shared_ptr<TensorImpl> Conv3d::forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) {
        
       
        try {
            const auto& in_tensor = operands[0];
            const auto& src_shape = in_tensor->shape();
            const auto& weights = operands[1];
            const auto& weights_shape = weights->shape();
            const auto& bias = operands.size() > 2 ? operands[2] : nullptr;

            if (
                src_shape.size() != 5 || 
                weights_shape.size()   != 5 ||
                bias && bias->shape().size() != 1
            ){
                throw std::invalid_argument(" in_tensor and weights must be of shape (B,C,D,H,W) while bias must be 1d tensor ");
            }

            if (
                src_shape[1] != weights_shape[1] 
            ){
                throw std::invalid_argument(" in_tensor and weights must have same channels number (if in_tensor is (B,C,D,H,W) then weights must be (OC,C,D,KH,KW)) ! ");
            }

            if (
                bias && bias->shape()[0] != weights_shape[0] 
            ){
                throw std::invalid_argument(" bias and weights must have same out_channels number (if bias is (OC) then weights must be (OC,C,D,KH,KW)) ! ");
            }

            dnnl::engine eng(dnnl::engine::kind::cpu, 0);
            dnnl::stream strm(eng);

            const auto OD = (src_shape[2] -  weights_shape[2] + m_padding_l[0] +  m_padding_r[0]) / m_strides[0] + 1;
            const auto OH = (src_shape[3] -  weights_shape[3] + m_padding_l[1] +  m_padding_r[1]) / m_strides[1] + 1;
            const auto OW = (src_shape[4] -  weights_shape[4] + m_padding_l[2] +  m_padding_r[2]) / m_strides[2] + 1;

            dnnl::memory::dims dst_dims = {src_shape[0], weights_shape[0], OD, OH, OW};

            const auto dst_strides = row_major_stride(dst_dims);
            
            dnnl::convolution_forward::primitive_desc fwd_conv_pd;

            auto dst_data = custom_conv_op_forward(
                in_tensor,
                weights,
                bias,
                dst_dims,
                dst_strides,
                m_strides,
                m_padding_l,
                m_padding_r,
                eng,
                strm,
                fwd_conv_pd
            );

            
            
            std::shared_ptr<Conv3d> grad_fn = nullptr;
            bool requires_grad = false;

            if ( in_tensor->requires_grad() ){
                requires_grad = true; 
                grad_fn = std::make_shared<Conv3d>(
                    m_strides,
                    m_padding_l,
                    m_padding_r,
                    true
                );   
                if (bias)
                    grad_fn->set_operands({in_tensor, weights, bias}); 
                else
                    grad_fn->set_operands({in_tensor, weights}); 
            }

            return std::make_shared<TensorImpl>(dst_data , 0 , dst_dims , grad_fn , requires_grad, true , dst_strides);

        }catch(std::exception& e){

            throw std::invalid_argument(std::string("error: Conv3d() was not possible for in_tensor: ") + e.what());

        }
    }  

    void Conv3d::backward(const std::shared_ptr<TensorImpl>& diff_loss_out){
        
        dnnl::engine engine(dnnl::engine::kind::cpu, 0);
        dnnl::stream engine_stream(engine);

        const auto& x = m_operands[0];
        const auto& w = m_operands[1];
        const auto& b = m_operands.size() > 2 ? m_operands[2] : nullptr;

        auto diff_dst_md = dnnl::memory::desc(diff_loss_out->shape() , dnnl::memory::data_type::f32, diff_loss_out->stride());

        if (x->requires_grad()){
            
        auto diff_src_md = dnnl::memory::desc(x->shape() , dnnl::memory::data_type::f32, row_major_stride(x->shape()));
            auto w_md = dnnl::memory::desc(w->shape() , dnnl::memory::data_type::f32, w->stride());

            auto bwd_pd = dnnl::convolution_backward_data::primitive_desc(
                engine,
                dnnl::algorithm::convolution_direct,
                diff_src_md,
                w_md,
                diff_dst_md,
                m_strides,
                m_padding_l,
                m_padding_r,
                m_conv_fwd_pd
            );

            dnnl::memory diff_dst_mem(diff_dst_md, engine, diff_loss_out->data_ptr().get() + diff_loss_out->data_offset());
            dnnl::memory w_mem(w_md, engine, w->data_ptr().get() + w->data_offset());
            dnnl::memory diff_src_mem;
            std::shared_ptr<float> data_storage; 
            
            if (x->get_grad()){
                diff_src_mem = dnnl::memory(diff_src_md, engine); 
            }else{
                data_storage = std::shared_ptr<float>(new float[x->numel()], std::default_delete<float[]>());
                diff_src_mem = dnnl::memory(diff_src_md, engine, data_storage.get()); 
            }


            auto conv_bwd = dnnl::convolution_backward_data(bwd_pd);
            conv_bwd.execute(engine_stream, {
                {DNNL_ARG_DIFF_DST, diff_dst_mem},
                {DNNL_ARG_DIFF_SRC, diff_src_mem},
                {DNNL_ARG_WEIGHTS, w_mem}
            });

            engine_stream.wait();

            if (x->get_grad()){
                accumulate(
                    diff_src_mem,
                    x->get_grad(),
                    engine,
                    engine_stream
                );
            }else {
                x->set_grad(std::make_shared<TensorImpl>(data_storage, 0 , diff_src_md.get_dims(), nullptr , false, true , diff_src_md.get_strides()));
            }
        }


        
        if (w->requires_grad() || b && b->requires_grad()){

            auto x_md = dnnl::memory::desc(x->shape() , dnnl::memory::data_type::f32, x->stride());
            auto diff_w_md = dnnl::memory::desc(w->shape() , dnnl::memory::data_type::f32, row_major_stride(w->shape()));
            auto diff_b_md = b
                ? dnnl::memory::desc( b->shape(), dnnl::memory::data_type::f32, row_major_stride(b->shape()))
                : dnnl::memory::desc();

            auto bwd_pd = dnnl::convolution_backward_weights::primitive_desc(
                engine,
                dnnl::algorithm::convolution_direct,
                x_md,
                diff_w_md,
                diff_b_md,
                diff_dst_md,
                m_strides,
                m_padding_l,
                m_padding_r ,
                m_conv_fwd_pd
            );

            dnnl::memory diff_dst_mem(diff_dst_md, engine, diff_loss_out->data_ptr().get() + diff_loss_out->data_offset());
            dnnl::memory x_mem(x_md, engine, x->data_ptr().get() + x->data_offset());
            dnnl::memory diff_w_mem;
            std::shared_ptr<float> w_data_storage; 
            dnnl::memory diff_b_mem;
            std::shared_ptr<float> b_data_storage; 
            
            if (w->get_grad()){
                diff_w_mem = dnnl::memory(diff_w_md, engine); //x_grad exists so we allocate new temporary memory
            }else{
                w_data_storage = std::shared_ptr<float>(new float[w->numel()], std::default_delete<float[]>());
                diff_w_mem = dnnl::memory(diff_w_md, engine, w_data_storage.get()); 
            }

            if (b){
                if (b->get_grad()){
                    diff_b_mem = dnnl::memory(diff_b_md, engine); 
                }else{
                    b_data_storage = std::shared_ptr<float>(new float[b->numel()], std::default_delete<float[]>());
                    diff_b_mem = dnnl::memory(diff_b_md, engine, b_data_storage.get()); 
                }
            }

            auto conv_bwd = dnnl::convolution_backward_weights(bwd_pd);
            conv_bwd.execute(engine_stream, {
                {DNNL_ARG_DIFF_DST, diff_dst_mem},
                {DNNL_ARG_SRC, x_mem},
                {DNNL_ARG_DIFF_WEIGHTS, diff_w_mem},
                {DNNL_ARG_DIFF_BIAS, diff_b_mem}
            });

            engine_stream.wait();

            if (w->get_grad()){
                accumulate(
                    diff_w_mem,
                    w->get_grad(),
                    engine,
                    engine_stream
                );
            }else {
                w->set_grad(std::make_shared<TensorImpl>(w_data_storage, 0 , diff_w_md.get_dims(), nullptr , false, true , diff_w_md.get_strides()));
            }

            if (b){
                if (b->get_grad()){
                    accumulate(
                        diff_b_mem,
                        b->get_grad(),
                        engine,
                        engine_stream
                    );
                }else {
                    b->set_grad(std::make_shared<TensorImpl>(b_data_storage, 0 , diff_b_md.get_dims(), nullptr , false, true , diff_b_md.get_strides()));
                }
            }
            
        }
    }

}//ops
}//mt