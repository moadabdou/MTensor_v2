#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <MTensor/tensorImpl.hpp>
#include <MTensor/ops.hpp>
#include <MTensor/utils/braodcast.hpp>

namespace mt {
namespace ops{

    int64_t Matmul::count = 0;

    Matmul::Matmul(bool inc_counter )
    {
        if(inc_counter){
            m_name = "Matmul"+std::to_string(count);
            count++;
        }
    } 
 
    std::shared_ptr<TensorImpl> Matmul::forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) {
        
        try{
              
            if (operands.size() != 2 && operands.size() != 3){
                throw  std::invalid_argument(" invalide operand number be 2 or 3");
            }      
            
            ops::Contiguous contiguous;
  

            auto [in_tensor_0, in_tensor_1] = utils::broadcast_matmul({
                operands[0]->is_contiguous() ? operands[0] :  contiguous.forward({operands[0]}), 
                operands[1]->is_contiguous() ? operands[1] :  contiguous.forward({operands[1]})
            });

            const auto& in_tensor_0_shape = in_tensor_0->shape();
            const auto& in_tensor_1_shape = in_tensor_1->shape();
            const auto& in_tensor_0_stride = in_tensor_0->stride();
            const auto& in_tensor_1_stride = in_tensor_1->stride();

            auto dst_shape = in_tensor_0_shape;
            dst_shape[dst_shape.size() - 1] = in_tensor_1_shape[in_tensor_1_shape.size() - 1];

            const auto& in_tensor_2 = operands.size() != 2 ? 
                utils::broadcast_to_shape(
                    operands[2]->is_contiguous() ? operands[2] :  contiguous.forward({operands[2]})
                    , dst_shape
                ) : nullptr ; 

            const auto& dst_stride = mt::row_major_stride(dst_shape);
            const auto& dst_numel = dst_stride[0] * dst_shape[0];

            std::shared_ptr<float> dst_data(new float[dst_numel], std::default_delete<float[]>());

            dnnl::engine engine(dnnl::engine::kind::cpu , 0);
            dnnl::stream strm(engine);

            auto in_tensor_0_md = dnnl::memory::desc( in_tensor_0_shape , dnnl::memory::data_type::f32, in_tensor_0_stride);
            auto in_tensor_1_md = dnnl::memory::desc( in_tensor_1_shape , dnnl::memory::data_type::f32, in_tensor_1_stride);
            auto dst_md = dnnl::memory::desc( dst_shape , dnnl::memory::data_type::f32, dst_stride);


            auto in_tensor_0_mem = dnnl::memory(in_tensor_0_md, engine, in_tensor_0->data_ptr().get() + in_tensor_0->data_offset());
            auto in_tensor_1_mem = dnnl::memory(in_tensor_1_md, engine, in_tensor_1->data_ptr().get() + in_tensor_1->data_offset());
            auto dst_mem = dnnl::memory(dst_md, engine, dst_data.get());

            std::unordered_map<int, dnnl::memory> matmul_args;
            
            matmul_args.insert({DNNL_ARG_SRC, in_tensor_0_mem});
            matmul_args.insert({DNNL_ARG_WEIGHTS, in_tensor_1_mem});
            matmul_args.insert({DNNL_ARG_DST, dst_mem});

            if (in_tensor_2){

                auto in_tensor_2_md = dnnl::memory::desc( in_tensor_2->shape() , dnnl::memory::data_type::f32, in_tensor_2->stride());
                auto in_tensor_2_mem = dnnl::memory(in_tensor_2_md, engine, in_tensor_2->data_ptr().get() + in_tensor_2->data_offset());

                matmul_args.insert({DNNL_ARG_BIAS, in_tensor_2_mem});

                auto matmul_pd = dnnl::matmul::primitive_desc( engine, in_tensor_0_md , in_tensor_1_md, in_tensor_2_md, dst_md);
                
                auto matmul_prim = dnnl::matmul(matmul_pd);
                
                matmul_prim.execute(strm, matmul_args); 

            }else {

                auto matmul_pd = dnnl::matmul::primitive_desc( engine, in_tensor_0_md , in_tensor_1_md, dst_md);
                
                auto matmul_prim = dnnl::matmul(matmul_pd);
                
                matmul_prim.execute(strm, matmul_args);   

            }

            strm.wait();

            std::shared_ptr<Operation> grad_fn = nullptr;
            bool requires_grad = false;

            if ( 
                in_tensor_0->requires_grad() || 
                in_tensor_1->requires_grad() || 
                in_tensor_2 && in_tensor_2->requires_grad()
            ){
                requires_grad = true; 
                grad_fn = std::make_shared<Matmul>(true);
                if ( in_tensor_2 ){
                    grad_fn->set_operands({in_tensor_0, in_tensor_1, in_tensor_2});
                }else{
                    grad_fn->set_operands({in_tensor_0, in_tensor_1});
                }
            }

            return std::make_shared<TensorImpl>(dst_data , 0 , dst_shape , grad_fn , requires_grad, true , dst_stride);

        }catch(std::exception& e){
            throw std::runtime_error(
                std::string("error : Matmul() ") + e.what()
            );
        }

    }  

    void Matmul::backward(const std::shared_ptr<TensorImpl>& diff_loss_out){

    }

}//ops
}//mt