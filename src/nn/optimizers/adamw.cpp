#include <MTensor/nn.hpp>
#include <MTensor/kernels.hpp>

namespace mt {
namespace nn {
namespace optimizer{

    AdamW::AdamW(
        const std::vector<Tensor>& parameters,
        float lr , 
        float weight_decay,
        float beta1,
        float beta2,
        float eps 
    ):
    Optimizer(parameters, lr),
    m_weight_decay(weight_decay),
    m_beta1(beta1),
    m_beta2(beta2),
    m_eps(eps)
    {

       m.reserve(parameters.size());
       v.reserve(parameters.size());

       if (parameters.size() > 100){
            #pragma omp parallel for 
            for (int64_t i = 0; i < parameters.size() ; i++){
                    m.push_back(Tensor::zeros_like(parameters[i]));
                    v.push_back(Tensor::zeros_like(parameters[i]));
            }
       }else {
            for (int64_t i = 0; i < parameters.size() ; i++){
                    m.push_back(Tensor::zeros_like(parameters[i]));
                    v.push_back(Tensor::zeros_like(parameters[i]));
            }
       }


    }

    void AdamW::step() {

        for(int64_t i = 0; i < m_parameters.size(); i++){
            const Tensor& param = m_parameters[i];
            const Tensor& grad = param.grad();
           
            if (!grad.tensor_impl()) continue;

            int64_t size = param.numel();
            fused_adam_w_kernel_avx512(
                param.data_ptr() + param.data_offset(),
                grad.data_ptr() + grad.data_offset(),
                m[i].data_ptr(),
                v[i].data_ptr(),
                size,
                m_beta1,
                m_beta2,
                &m_beta1_t,
                &m_beta2_t,
                m_eps,
                m_lr,
                m_weight_decay
            );
        }

    }

    std::vector<Tensor>& AdamW::stats_m(){
        return m;
    }
    std::vector<Tensor>& AdamW::stats_v(){
        return v;
    }


}//optimizer

}//nn
}//mt
