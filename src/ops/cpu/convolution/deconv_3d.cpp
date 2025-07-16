#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <MTensor/tensorImpl.hpp>
#include <MTensor/ops.hpp>

namespace mt {
namespace ops{

    int64_t Deconv3d::count = 0;

    Deconv3d::Deconv3d(
        const std::vector<int64_t>& strides,
        const std::vector<int64_t>& padding_l,
        const std::vector<int64_t>& padding_r,
        bool inc_counter)
    {

        if (strides.size() != 3 || strides[0] < 0 || strides[1] < 0 || strides[2] < 0){
            throw std::invalid_argument("error : Deconv3d() invalid strides were given");
        }
        if (padding_l.size() != 3 || padding_l[0] < 0 || padding_l[1] < 0 || padding_l[2] < 0){
            throw std::invalid_argument("error : Deconv3d() invalid padding_l was given");
        }
        if (padding_r.size() != 3 || padding_r[0] < 0 || padding_r[1] < 0 || padding_r[2] < 0 ){
            throw std::invalid_argument("error : Deconv3d() invalid padding_r was given");
        }

        m_strides = strides;
        m_padding_l = padding_l;
        m_padding_r = padding_r;

        if(inc_counter){
            m_name = "Deconv3d"+std::to_string(count);
            count++;
        }
    } 
 
    std::shared_ptr<TensorImpl> Deconv3d::forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) {
        
       
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

            const auto OD = (src_shape[2] - 1 ) * m_strides[0] - m_padding_l[0] -  m_padding_r[0] + weights_shape[2];
            const auto OH = (src_shape[3] - 1 ) * m_strides[1] - m_padding_l[1] -  m_padding_r[1] + weights_shape[3];
            const auto OW = (src_shape[4] - 1 ) * m_strides[2] - m_padding_l[2] -  m_padding_r[2] + weights_shape[4];

            dnnl::memory::dims dst_dims = {src_shape[0], weights_shape[0], OD, OH, OW};

            const auto dst_strides = row_major_stride(dst_dims);
            
            auto dst_data = custom_deconv_op_forward(
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
                m_fwd_deconv_pd
            );

            
            
            std::shared_ptr<Deconv3d> grad_fn = nullptr;
            bool requires_grad = false;

            if ( in_tensor->requires_grad() || weights->requires_grad() || bias->requires_grad()){
                requires_grad = true; 
                grad_fn = std::make_shared<Deconv3d>(
                    m_strides,
                    m_padding_l,
                    m_padding_r,
                    true
                );
                grad_fn->m_fwd_deconv_pd = m_fwd_deconv_pd; 
                if (bias)
                    grad_fn->set_operands({in_tensor, weights, bias}); 
                else
                    grad_fn->set_operands({in_tensor, weights}); 
            }

            return std::make_shared<TensorImpl>(dst_data , 0 , dst_dims , grad_fn , requires_grad, true , dst_strides);

        }catch(std::exception& e){

            throw std::invalid_argument(std::string("error: Deconv3d() was not possible for in_tensor: ") + e.what());

        }
    }  

    void Deconv3d::backward(const std::shared_ptr<TensorImpl>& diff_loss_out){

        dnnl::engine engine(dnnl::engine::kind::cpu, 0);
        dnnl::stream engine_stream(engine);

        const auto &x = m_operands[0];
        const auto &w = m_operands[1];
        const auto &b = m_operands.size() > 2 ? m_operands[2] : nullptr;

        deconv_backward(
            engine,
            engine_stream,
            x,w,b,
            diff_loss_out,
            m_fwd_deconv_pd,
            m_strides,
            m_padding_l,
            m_padding_r
        );

    }

}//ops
}//mt