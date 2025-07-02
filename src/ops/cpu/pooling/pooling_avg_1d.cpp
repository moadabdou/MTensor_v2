#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <MTensor/tensorImpl.hpp>
#include <MTensor/ops.hpp>

namespace mt {
namespace ops{

    int64_t AvgPooling1d::count = 0;

    AvgPooling1d::AvgPooling1d(
        const std::vector<int64_t>& kernel,
        const std::vector<int64_t>& strides,
        const std::vector<int64_t>& padding_l,
        const std::vector<int64_t>& padding_r,
        bool include_padding, 
        bool inc_counter):
    m_include_padding(include_padding)
    {
        if (kernel.size() != 1 || kernel[0] < 0){
            throw std::invalid_argument("error : AvgPooling1d() invalid kernel was given");
        }
        if (strides.size() != 1 || strides[0] < 0){
            throw std::invalid_argument("error : AvgPooling1d() invalid strides were given");
        }
        if (padding_l.size() != 1 || padding_l[0] < 0){
            throw std::invalid_argument("error : AvgPooling1d() invalid padding_l was given");
        }
        if (padding_r.size() != 1 || padding_r[0] < 0){
            throw std::invalid_argument("error : AvgPooling1d() invalid padding_r was given");
        }
        m_kernel = kernel;
        m_strides = strides;
        m_padding_l = padding_l;
        m_padding_r = padding_r;

        if(inc_counter){
            m_name = "AvgPooling1d"+std::to_string(count);
            count++;
        }
    } 
 
    std::shared_ptr<TensorImpl> AvgPooling1d::forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) {
        
       
        try {
            const auto& in_tensor = operands[0];

            if (in_tensor->shape().size() != 3 ){
                throw std::invalid_argument(" in_tensor must be of shape (B,C,T) ");
            }

            std::unique_ptr<dnnl::memory> workspace_mem = nullptr;

            dnnl::engine eng(dnnl::engine::kind::cpu, 0);
            dnnl::stream strm(eng);

            dnnl::pooling_forward::primitive_desc pool_fwd_pd;


            const  auto& src_shape = in_tensor->shape();
            dnnl::memory::dims dst_dims = src_shape;
            dst_dims[2] = (src_shape[2] -  m_kernel[0] + m_padding_l[0] +  m_padding_r[0]) / m_strides[0] + 1;

            auto dst_data = custom_pooling_op_forward(
                in_tensor,
                m_include_padding ? dnnl::algorithm::pooling_avg_include_padding : dnnl::algorithm::pooling_avg_exclude_padding,
                dst_dims,
                m_kernel,
                m_strides,
                m_padding_l,
                m_padding_r,
                eng,
                strm,
                pool_fwd_pd,
                false,
                workspace_mem
            );

            const auto& dst_md = pool_fwd_pd.dst_desc();
            std::shared_ptr<AvgPooling1d> grad_fn = nullptr;
            bool requires_grad = false;

            if ( in_tensor->requires_grad() ){
                requires_grad = true; 
                grad_fn = std::make_shared<AvgPooling1d>(
                    m_kernel,
                    m_strides,
                    m_padding_l,
                    m_padding_r,
                    m_include_padding,
                    true
                );   
                grad_fn->set_operands({in_tensor}); 
            }

            return std::make_shared<TensorImpl>(dst_data , 0 , dst_md.get_dims() , grad_fn , requires_grad, true , dst_md.get_strides());

        }catch(std::exception& e){

            throw std::invalid_argument(std::string("error: AvgPooling1d() was not possible for in_tensor: ") + e.what());

        }
    }  

    void AvgPooling1d::backward(const std::shared_ptr<TensorImpl>& diff_loss_out){
        
    }



}//ops
}//mt