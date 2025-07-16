#include <stdexcept>
#include <numeric>
#include <iostream>
#include <omp.h>
#include <MTensor/tensorImpl.hpp>
#include <MTensor/ops.hpp>



namespace mt {
namespace ops{

    int64_t Embedding::count = 0;

    Embedding::Embedding(const std::vector<int64_t>& indices , bool inc_counter )
    {
        if ( 0 > *std::min_element(indices.begin(), indices.end())){
            throw std::invalid_argument(
                "error: Embedding() negative indice are not allowed"
            );
        }
        m_indices = indices;
        if(inc_counter){
            m_name = "Embedding"+std::to_string(count);
            count++;
        }
    }
 
    std::shared_ptr<TensorImpl> Embedding::forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) {

        const auto& in_tensor = operands[0];
        const auto& in_shape = in_tensor->shape();
        const auto& in_stride = in_tensor->stride();

        if (*std::max_element(m_indices.begin(),m_indices.end()) >= in_shape[0]) {
            throw std::invalid_argument(
                "error: Embedding() indices are out of range of the in_tensor's shape first dim"
            );
        }

        if (!in_tensor->is_contiguous()){
            throw std::invalid_argument(
                "error: Embedding() does not support sparse tensors"
            );
        }

        auto out_shape = in_shape;
        out_shape[0] = m_indices.size();
        auto sub_shape = in_shape;
        sub_shape[0] = 1;


        const auto& out_stride = row_major_stride(out_shape);

        const int64_t out_numel = out_shape[0] * out_stride[0];

        std::shared_ptr<float> out_data(new float[out_numel], std::default_delete<float[]>());

        
        dnnl::engine eng(dnnl::engine::kind::cpu, 0);
        dnnl::stream strm(eng);
        #pragma omp parallel for
        for (int64_t i = 0; i < m_indices.size(); i++){
 
            dnnl::memory::desc src_md(sub_shape , dnnl::memory::data_type::f32, in_stride);
            dnnl::memory::desc dst_md(sub_shape , dnnl::memory::data_type::f32, out_stride);

            dnnl::memory src_m(src_md, eng, in_tensor->data_ptr().get() + in_tensor->data_offset() + m_indices[i]*in_stride[0]);
            dnnl::memory dst_m(dst_md, eng, out_data.get() + i * out_stride[0]);

            dnnl::reorder reoder_primitive(src_m, dst_m);

            reoder_primitive.execute(strm, src_m, dst_m);
            strm.wait();     

        }

        std::shared_ptr<Operation> grad_fn = nullptr;
        bool requires_grad = false;

        if (in_tensor->requires_grad()){ 
            requires_grad = true; 
            grad_fn = std::make_shared<Embedding>(m_indices, true);
            grad_fn->set_operands({in_tensor});
        }

        return std::make_shared<TensorImpl>(out_data, 0, out_shape , grad_fn , requires_grad, true , out_stride);
    }  

    void Embedding::backward(const std::shared_ptr<TensorImpl>& diff_loss_out){
//if (diff_loss_out->requires_grad()) std::cout << m_name ; 
        dnnl::engine engine(dnnl::engine::kind::cpu, 0);

        const auto& in_tensor = m_operands[0];

        if (!in_tensor->requires_grad()) return;

        
        {

        if (!in_tensor->get_grad()){
            in_tensor->set_grad(TensorImpl::zeros(in_tensor->shape()));
        }
        
        auto sub_shape = in_tensor->shape();
        sub_shape[0] = 1;

        auto in_grad_stride = in_tensor->get_grad()->stride();
        auto in_grad_data_ptr = in_tensor->get_grad()->data_ptr().get();

        auto out_diff_stride = diff_loss_out->stride();
        auto out_diff_data_ptr = diff_loss_out->data_ptr().get();
        auto out_diff_data_offset = diff_loss_out->data_offset();

        #pragma omp parallel for
        for (int64_t i = 0 ; i < m_indices.size() ;  i++){

            dnnl::stream engine_stream(engine);
            dnnl::memory::desc diff_src_md(sub_shape , dnnl::memory::data_type::f32, in_grad_stride);
            dnnl::memory::desc diff_dst_md(sub_shape , dnnl::memory::data_type::f32, diff_loss_out->stride());

            dnnl::memory diff_src_m(diff_src_md, engine, in_grad_data_ptr + m_indices[i]*in_grad_stride[0]);
            dnnl::memory diff_dst_m(diff_dst_md, engine, out_diff_data_ptr + out_diff_data_offset + i *out_diff_stride[0]);
            accumulate_mem(
                diff_dst_m,
                diff_src_m,
                engine,
                engine_stream
            );
        }

        }

    }

}//ops
}//mt