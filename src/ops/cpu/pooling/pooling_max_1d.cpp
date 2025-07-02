#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <MTensor/tensorImpl.hpp>
#include <MTensor/ops.hpp>

namespace mt {
namespace ops{

    int64_t MaxPooling1d::count = 0;

    MaxPooling1d::MaxPooling1d(
        const std::vector<int64_t>& kernel,
        const std::vector<int64_t>& strides,
        const std::vector<int64_t>& padding_l,
        const std::vector<int64_t>& padding_r, 
        bool inc_counter)
    {
        if (kernel.size() != 1 || kernel[0] < 0){
            throw std::invalid_argument("error : MaxPooling1d() invalid kernel was given");
        }
        if (strides.size() != 1 || strides[0] < 0){
            throw std::invalid_argument("error : MaxPooling1d() invalid strides were given");
        }
        if (padding_l.size() != 1 || padding_l[0] < 0){
            throw std::invalid_argument("error : MaxPooling1d() invalid padding_l was given");
        }
        if (padding_r.size() != 1 || padding_r[0] < 0){
            throw std::invalid_argument("error : MaxPooling1d() invalid padding_r was given");
        }
        m_kernel = kernel;
        m_strides = strides;
        m_padding_l = padding_l;
        m_padding_r = padding_r;

        if(inc_counter){
            m_name = "MaxPooling1d"+std::to_string(count);
            count++;
        }
    } 
 
    std::shared_ptr<TensorImpl> MaxPooling1d::forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) {
        
       
        try {
            const auto& in_tensor = operands[0];

            if (in_tensor->shape().size() != 3 ){
                throw std::invalid_argument(" in_tensor must be of shape (B,C,T) ");
            }

            std::unique_ptr<dnnl::memory> workspace_mem;

            dnnl::engine eng(dnnl::engine::kind::cpu, 0);
            dnnl::stream strm(eng);

            dnnl::pooling_forward::primitive_desc pool_fwd_pd;


            const  auto& src_shape = in_tensor->shape();
            dnnl::memory::dims dst_dims = src_shape;
            dst_dims[2] = (src_shape[2] -  m_kernel[0] + m_padding_l[0] +  m_padding_r[0]) / m_strides[0] + 1;

            auto dst_data = custom_pooling_op_forward(
                in_tensor,
                dnnl::algorithm::pooling_max,
                dst_dims,
                m_kernel,
                m_strides,
                m_padding_l,
                m_padding_r,
                eng,
                strm,
                pool_fwd_pd,
                in_tensor->requires_grad(),
                workspace_mem
            );
            
            const auto& dst_md = pool_fwd_pd.dst_desc();
            std::shared_ptr<MaxPooling1d> grad_fn = nullptr;
            bool requires_grad = false;

            if ( in_tensor->requires_grad() ){
                requires_grad = true; 
                grad_fn = std::make_shared<MaxPooling1d>(
                    m_kernel,
                    m_strides,
                    m_padding_l,
                    m_padding_r,
                    true
                );
                grad_fn->m_workspace_mem = std::move(workspace_mem);   
                grad_fn->set_operands({in_tensor}); 
            }

            return std::make_shared<TensorImpl>(dst_data , 0 , dst_md.get_dims() , grad_fn , requires_grad, true , dst_md.get_strides());

        }catch(std::exception& e){

            throw std::invalid_argument(std::string("error: MaxPooling1d() was not possible for in_tensor: ") + e.what());

        }
    }  

    void MaxPooling1d::backward(const std::shared_ptr<TensorImpl>& diff_loss_out){
        utils::print_vector(m_workspace_mem->get_desc().get_dims());
    }



}//ops
}//mt