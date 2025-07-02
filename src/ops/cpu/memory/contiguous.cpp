#include <stdexcept>
#include <iostream>
#include <MTensor/tensorImpl.hpp>
#include <MTensor/ops.hpp>

namespace mt {
namespace ops{

    int64_t Contiguous::count = 0;

    Contiguous::Contiguous(bool inc_counter )
    {
        if(inc_counter){
            m_name = "Contiguous"+std::to_string(count);
            count++;
        }
    } 
 
    std::shared_ptr<TensorImpl> Contiguous::forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) {

        const auto& in_tensor = operands[0];

        std::shared_ptr<float> data_storage(new float[in_tensor->numel()], std::default_delete<float[]>());

        const auto& in_shape = in_tensor->shape();
        const auto& out_stride = row_major_stride(in_shape);

        dnnl::engine eng (dnnl::engine::kind::cpu,0);
        dnnl::stream strm(eng);

        dnnl::memory::desc src_md(in_shape, dnnl::memory::data_type::f32, in_tensor->stride());
        dnnl::memory::desc dst_md(in_shape, dnnl::memory::data_type::f32, out_stride);

        dnnl::memory src_m(src_md, eng, in_tensor->data_ptr().get() + in_tensor->data_offset());
        dnnl::memory dst_m(dst_md, eng, data_storage.get());

        dnnl::reorder reorder_prm(src_m, dst_m);
        reorder_prm.execute(strm, src_m, dst_m);

        strm.wait();

        std::shared_ptr<Operation> grad_fn = nullptr;
        bool requires_grad = false;

        if ( in_tensor->requires_grad() ){
            requires_grad = true; 
            grad_fn = std::make_shared<Contiguous>(true);
            grad_fn->set_operands({in_tensor});
        }

        return std::make_shared<TensorImpl>(data_storage , 0 , in_shape , grad_fn , requires_grad, true , out_stride);
    }  

    void Contiguous::backward(const std::shared_ptr<TensorImpl>& diff_loss_out){

    }

}//ops
}//mt