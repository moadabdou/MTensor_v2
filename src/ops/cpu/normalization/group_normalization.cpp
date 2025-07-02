#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <MTensor/tensorImpl.hpp>
#include <MTensor/ops.hpp>


namespace mt {
namespace ops{

    int64_t GroupNormalization::count = 0;

    GroupNormalization::GroupNormalization(
        int64_t groups,
        bool inc_counter
    ): m_groups(groups){

        if(inc_counter){
            m_name = "GroupNormalization"+std::to_string(count);
            count++;
        }

    } 
 
    std::shared_ptr<TensorImpl> GroupNormalization::forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) {
        
        try {
            const auto& in_tensor = operands[0];
            const auto& src_shape = in_tensor->shape();
            const auto& scale = operands[1];
            const auto& shift = operands[2];
            const auto& scale_shift_shape = scale->shape();
            const auto dst_stride =  row_major_stride(src_shape);

            if (
                src_shape.size() < 2 ||
                scale_shift_shape.size() != 1 ||
                scale_shift_shape != shift->shape()
            ){
                throw std::invalid_argument(" in_tensor must be 2D, higher while scale & shift must be 1D contiguous tensors with same shape ");
            }

            if (
                ! scale->is_contiguous() || 
                ! shift->is_contiguous()
            ){
                throw std::invalid_argument(" scale & shift must be contiguous ");
            }

            if (
                m_groups > src_shape[1] ||
                src_shape[1] % m_groups != 0
            ){
                throw std::invalid_argument(" in_tensor channels must be equal or bigger than normalization groups and divisible by it (C %% G = 0)");
            }

            dnnl::engine eng(dnnl::engine::kind::cpu, 0);
            dnnl::stream strm(eng);

            
            std::shared_ptr<float> dst_data(new float[src_shape[0] * dst_stride[0]], std::default_delete<float[]>());

            auto src_md = dnnl::memory::desc(
                        src_shape, dnnl::memory::data_type::f32,  in_tensor->stride());
            auto dst_md = dnnl::memory::desc(
                    src_shape, dnnl::memory::data_type::f32, dst_stride);
            auto scaleshift_md = dnnl::memory::desc(
                    scale_shift_shape, dnnl::memory::data_type::f32, dnnl::memory::format_tag::x);
            
            
            auto src_mem = dnnl::memory(src_md, eng, in_tensor->data_ptr().get() + in_tensor->data_offset());
            auto dst_mem = dnnl::memory(dst_md, eng, dst_data.get());
            auto scale_mem = dnnl::memory(scaleshift_md, eng, scale->data_ptr().get() + scale->data_offset());
            auto shift_mem = dnnl::memory(scaleshift_md, eng, shift->data_ptr().get() + shift->data_offset());


            // Create primitive descriptor.
            auto gnorm_pd = dnnl::group_normalization_forward::primitive_desc(eng,
                dnnl::prop_kind::forward_training, src_md, dst_md, m_groups, 1.e-10f,
                dnnl::normalization_flags::use_scale | dnnl::normalization_flags::use_shift);
            

            std::unique_ptr<dnnl::memory> mean_mem (new dnnl::memory(gnorm_pd.mean_desc(), eng));
            std::unique_ptr<dnnl::memory> variance_mem (new dnnl::memory(gnorm_pd.variance_desc(), eng));

            
            // Create the primitive.
            auto gnorm_prim = dnnl::group_normalization_forward(gnorm_pd);

            // Primitive arguments. Set up in-place execution by assigning src as DST.
            std::unordered_map<int, dnnl::memory> gnorm_args;
            gnorm_args.insert({DNNL_ARG_SRC, src_mem});
            gnorm_args.insert({DNNL_ARG_MEAN, *mean_mem});
            gnorm_args.insert({DNNL_ARG_VARIANCE, *variance_mem});
            gnorm_args.insert({DNNL_ARG_SCALE, scale_mem});
            gnorm_args.insert({DNNL_ARG_SHIFT, shift_mem});
            gnorm_args.insert({DNNL_ARG_DST, dst_mem});
            // Primitive execution: batch normalization with ReLU.
            gnorm_prim.execute(strm, gnorm_args);

            // Wait for the computation to finalize.
            strm.wait();

            std::shared_ptr<GroupNormalization> grad_fn = nullptr;
            bool requires_grad = false;

            if ( in_tensor->requires_grad() || shift->requires_grad() || scale->requires_grad()){
                requires_grad = true; 
                grad_fn = std::make_shared<GroupNormalization>(m_groups, true);   
                grad_fn->set_operands({in_tensor, scale, shift}); 
                grad_fn->m_mean = std::move(mean_mem);
                grad_fn->m_variance = std::move(variance_mem);

            }

            return std::make_shared<TensorImpl>(dst_data , 0 , src_shape , grad_fn , requires_grad, true , dst_stride);

        }catch(std::exception& e){

            throw std::invalid_argument(std::string("error: GroupNormalization() was not possible for in_tensor: ") + e.what());

        }
    }  

    void GroupNormalization::backward(const std::shared_ptr<TensorImpl>& diff_loss_out){
        
    }



}//ops
}//mt