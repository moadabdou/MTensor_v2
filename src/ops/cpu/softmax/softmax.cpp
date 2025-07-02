#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <MTensor/tensorImpl.hpp>
#include <MTensor/ops.hpp>

namespace mt {
namespace ops{

    int64_t Softmax::count = 0;

    Softmax::Softmax(int dim , bool inc_counter)
    {
        if (dim < 0){
            throw std::invalid_argument("error : Softmax() invalid dim was given");
        }

        m_dim = dim;

        if(inc_counter){
            m_name = "Softmax"+std::to_string(count);
            count++;
        }
    } 
 
    std::shared_ptr<TensorImpl> Softmax::forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) {
        

        const auto& in_tensor = operands[0];


        const auto& in_shape = in_tensor->shape();
        const auto& src_stride = in_tensor->stride();

        if (m_dim >= in_shape.size()){
            throw std::invalid_argument("error: Softmax() dim is out of range of in_tensor dims");
        }

        const auto& dst_stride = row_major_stride(in_shape);

        std::shared_ptr<float> dst_data(new float[ dst_stride[0] * in_shape[0] ], std::default_delete<float[]>());
        
        dnnl::engine eng(dnnl::engine::kind::cpu, 0);
        dnnl::stream strm(eng);

        dnnl::memory::desc src_md(in_shape, dnnl::memory::data_type::f32, src_stride);
        dnnl::memory::desc dst_md(in_shape, dnnl::memory::data_type::f32, dst_stride);

        dnnl::memory src_m(src_md, eng, in_tensor->data_ptr().get() + in_tensor->data_offset());
        dnnl::memory dst_m(dst_md, eng, dst_data.get());

        dnnl::softmax_forward::primitive_desc softmax_desc(
            eng,  
            dnnl::prop_kind::forward_inference,
            dnnl::algorithm::softmax_accurate, 
            src_md, 
            dst_md,
            m_dim
        );

        auto softmax = dnnl::softmax_forward(softmax_desc); 

        std::unordered_map<int, dnnl::memory> softmax_args;
        softmax_args.insert({DNNL_ARG_SRC, src_m});
        softmax_args.insert({DNNL_ARG_DST, dst_m});

        softmax.execute(strm, softmax_args);

        strm.wait();

        std::shared_ptr<Operation> grad_fn = nullptr;
        bool requires_grad = false;

        if ( in_tensor->requires_grad() ){
            requires_grad = true; 
            grad_fn = std::make_shared<Softmax>(m_dim, true);
            grad_fn->set_operands({in_tensor});
        }

        return std::make_shared<TensorImpl>(dst_data , 0 , in_shape , grad_fn , requires_grad, true , dst_stride);
    }  

    void Softmax::backward(const std::shared_ptr<TensorImpl>& diff_loss_out){

    }

}//ops
}//mt