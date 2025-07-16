#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <MTensor/tensorImpl.hpp>
#include <MTensor/ops.hpp>

namespace mt {
namespace ops{

    int64_t MaxPooling3d::count = 0;

    MaxPooling3d::MaxPooling3d(
        const std::vector<int64_t>& kernel,
        const std::vector<int64_t>& strides,
        const std::vector<int64_t>& padding_l,
        const std::vector<int64_t>& padding_r, 
        bool inc_counter)
    {
        if (kernel.size() != 3 || kernel[0] < 0 || kernel[1] < 0 || kernel[2] < 0 ){
            throw std::invalid_argument("error : MaxPooling3d() invalid kernel was given");
        }
        if (strides.size() != 3 || strides[0] < 0 || strides[1] < 0 || strides[2] < 0){
            throw std::invalid_argument("error : MaxPooling3d() invalid strides were given");
        }
        if (padding_l.size() != 3 || padding_l[0] < 0 || padding_l[1] < 0 || padding_l[2] < 0){
            throw std::invalid_argument("error : MaxPooling3d() invalid padding_l was given");
        }
        if (padding_r.size() != 3 || padding_r[0] < 0 || padding_r[1] < 0 || padding_r[2] < 0){
            throw std::invalid_argument("error : MaxPooling3d() invalid padding_r was given");
        }
        m_kernel = kernel;
        m_strides = strides;
        m_padding_l = padding_l;
        m_padding_r = padding_r;

        if(inc_counter){
            m_name = "MaxPooling3d"+std::to_string(count);
            count++;
        }
    } 
 
    std::shared_ptr<TensorImpl> MaxPooling3d::forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) {
        
       
        try {
            const auto& in_tensor = operands[0];

            if (in_tensor->shape().size() != 5 ){
                throw std::invalid_argument(" in_tensor must be of shape (B,T,C,H,W) ");
            }

            std::unique_ptr<dnnl::memory> workspace_mem;

            dnnl::engine eng(dnnl::engine::kind::cpu, 0);
            dnnl::stream strm(eng);

            dnnl::pooling_forward::primitive_desc pool_fwd_pd;


            const  auto& src_shape = in_tensor->shape();
            dnnl::memory::dims dst_dims = src_shape;
            dst_dims[2] = (src_shape[2] -  m_kernel[0] + m_padding_l[0] +  m_padding_r[0]) / m_strides[0] + 1;
            dst_dims[3] = (src_shape[3] -  m_kernel[1] + m_padding_l[1] +  m_padding_r[1]) / m_strides[1] + 1;
            dst_dims[4] = (src_shape[4] -  m_kernel[2] + m_padding_l[2] +  m_padding_r[2]) / m_strides[2] + 1;
            

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
            std::shared_ptr<MaxPooling3d> grad_fn = nullptr;
            bool requires_grad = false;

            if ( in_tensor->requires_grad() ){
                requires_grad = true; 
                grad_fn = std::make_shared<MaxPooling3d>(
                    m_kernel,
                    m_strides,
                    m_padding_l,
                    m_padding_r,
                    true
                );
                grad_fn->m_workspace_mem = std::move(workspace_mem);   
                grad_fn->set_operands({in_tensor}); 
                grad_fn->m_pool_fwd_pd = pool_fwd_pd;
            }

            return std::make_shared<TensorImpl>(dst_data , 0 , dst_md.get_dims() , grad_fn , requires_grad, true , dst_md.get_strides());

        }catch(std::exception& e){

            throw std::invalid_argument(std::string("error: MaxPooling3d() was not possible for in_tensor: ") + e.what());

        }
    }  

    void MaxPooling3d::backward(const std::shared_ptr<TensorImpl>& diff_loss_out){
//if (diff_loss_out->requires_grad()) std::cout << m_name ; 
        
        dnnl::engine engine(dnnl::engine::kind::cpu, 0);
        dnnl::stream engine_stream(engine);

        const auto& x = m_operands[0];
        if (! x->requires_grad()) return;

        auto diff_src_md = dnnl::memory::desc(x->shape() , dnnl::memory::data_type::f32, row_major_stride(x->shape()));;
        auto diff_dst_md = dnnl::memory::desc(diff_loss_out->shape() , dnnl::memory::data_type::f32, diff_loss_out->stride());;

        // Backward primitive desc
        auto bwd_pd = dnnl::pooling_backward::primitive_desc(
            engine,
            dnnl::algorithm::pooling_max,
            diff_src_md,
            diff_dst_md,
            m_strides,
            m_kernel,
            {0,0,0},
            m_padding_l,
            m_padding_r,
            m_pool_fwd_pd
        );

        dnnl::memory diff_dst_mem(diff_dst_md, engine, diff_loss_out->data_ptr().get() + diff_loss_out->data_offset());
        dnnl::memory diff_src_mem;
        std::shared_ptr<float> data_storage; //in case if the x_grad does not exist
        
        if (x->get_grad()){
            diff_src_mem = dnnl::memory(diff_src_md, engine); //x_grad exists so we allocate new temporary memory
        }else{
            data_storage = std::shared_ptr<float>(new float[x->numel()], std::default_delete<float[]>());
            diff_src_mem = dnnl::memory(diff_src_md, engine, data_storage.get()); // x_grad does not exists so we make one and the result will directly routed to it 
        }


        auto pooling_bwd = dnnl::pooling_backward(bwd_pd);
        pooling_bwd.execute(engine_stream, {
            {DNNL_ARG_DIFF_DST, diff_dst_mem},
            {DNNL_ARG_DIFF_SRC, diff_src_mem},
            {DNNL_ARG_WORKSPACE, *m_workspace_mem}
        });

        engine_stream.wait();

        
        {
        if (x->get_grad())
        {
                accumulate(
                    diff_src_mem,
                    x->get_grad(),
                    engine,
                    engine_stream);
        }
        else
        {
            x->set_grad(std::make_shared<TensorImpl>(data_storage, 0, diff_src_md.get_dims(), nullptr, false, true, diff_src_md.get_strides()));
        }
        }
    }



}//ops
}//mt