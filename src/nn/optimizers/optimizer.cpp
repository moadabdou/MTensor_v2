#include <MTensor/nn.hpp>

namespace mt {
namespace nn {


void _make_zero(const Tensor& input){


    dnnl::engine engine(dnnl::engine::kind::cpu, 0);

    dnnl::stream engine_stream(engine);

    auto md = dnnl::memory::desc(
            input.shape(), dnnl::memory::data_type::f32, input.stride());

    auto mem = dnnl::memory(md, engine, input.data_ptr() + input.data_offset());

    auto linear_zero = dnnl::eltwise_forward::primitive_desc(
            engine,
            dnnl::prop_kind::forward_inference, 
            dnnl::algorithm::eltwise_linear , 
            md,
            md, 
            0.f, 
            0.f
        );

    auto linear_zero_prim = dnnl::eltwise_forward(linear_zero);
    std::unordered_map<int, dnnl::memory> eltwise_args;
    eltwise_args.insert({DNNL_ARG_SRC, mem});
    eltwise_args.insert({DNNL_ARG_DST, mem});
    linear_zero_prim.execute(engine_stream, eltwise_args);
    engine_stream.wait();

}


namespace optimizer{

    Optimizer::Optimizer(const std::vector<Tensor>& parameters, float lr):
    m_parameters(parameters), linear(0.0f, 0.0f), m_lr(lr)
    {}

    std::vector<Tensor>& Optimizer::paramters(){
        return m_parameters;
    }

    float Optimizer::get_lr() const{
        return m_lr;
    }

    void  Optimizer::set_lr(float lr){
        m_lr = lr;
    }

    void Optimizer::zero_grad(){
    
        if (m_parameters.size() > 100){
            #pragma omp parallel for 
            for (int64_t i = 0; i < m_parameters.size() ; i++){
                const Tensor& grad = m_parameters[i].grad();
                if (!grad.tensor_impl()) continue;
                _make_zero(grad);
            }
       }else {
            for (int64_t i = 0; i < m_parameters.size() ; i++){
                const Tensor& grad = m_parameters[i].grad();
                if (!grad.tensor_impl()) continue;
                _make_zero(grad);
            }
       }

    }

}//optimizer

}//nn
}//mt
