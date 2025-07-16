#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <MTensor/tensorImpl.hpp>
#include <MTensor/ops.hpp>
#include <MTensor/utils/braodcast.hpp>

namespace mt {
namespace ops{

    int64_t Ne::count = 0;

    Ne::Ne(bool inc_counter )
    {
        if(inc_counter){
            m_name = "Ne"+std::to_string(count);
            count++;
        }
    } 
 
    std::shared_ptr<TensorImpl> Ne::forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) {
        
        try{
                
            auto [in_tensor_0, in_tensor_1] = utils::broadcast({operands[0], operands[1]});

            const auto& in_shape = in_tensor_0->shape();
            const auto& src_0_stride = in_tensor_0->stride();
            const auto& src_1_stride = in_tensor_1->stride();
            const auto& dst_stride = row_major_stride(in_shape);

            dnnl::memory::desc src_0_md(in_shape, dnnl::memory::data_type::f32, src_0_stride);
            dnnl::memory::desc src_1_md(in_shape, dnnl::memory::data_type::f32, src_1_stride);
            dnnl::memory::desc dst_md(in_shape, dnnl::memory::data_type::f32, dst_stride);

            dnnl::engine eng(dnnl::engine::kind::cpu, 0);

            dnnl::binary::primitive_desc binary_desc(
                eng,
                dnnl::algorithm::binary_ne,
                src_0_md,
                src_1_md,
                dst_md
            );

            const auto& data_storage = custom_binary_op(
                in_tensor_0,
                in_tensor_1, 
                binary_desc, 
                eng, 
                src_0_md, 
                src_1_md, 
                dst_md
            );

            std::shared_ptr<Operation> grad_fn = nullptr;
            bool requires_grad = false;

            if ( in_tensor_0->requires_grad() || in_tensor_1->requires_grad()){
                requires_grad = true; 
                grad_fn = std::make_shared<Ne>(true);
                grad_fn->set_operands({in_tensor_0, in_tensor_1});
            }

            return std::make_shared<TensorImpl>(data_storage , 0 , in_shape , grad_fn , requires_grad, true , dst_stride);

        }catch(std::exception& e){
            throw std::runtime_error(
                std::string("error : Ne() ") + e.what()
            );
        }

    }  

    void Ne::backward(const std::shared_ptr<TensorImpl>& diff_loss_out){
//if (diff_loss_out->requires_grad()) std::cout << m_name ; 
        throw std::runtime_error("Backward is not implemented for Ne() '!=' ");
    }

}//ops
}//mt