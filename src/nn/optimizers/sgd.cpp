#include <MTensor/nn.hpp>
#include <MTensor/kernels.hpp>

namespace mt {
namespace nn {
namespace optimizer{

    SGD::SGD(
        const std::vector<Tensor>& parameters,
        float lr , 
        float weight_decay,
        float momentum
    ):
    Optimizer(parameters, lr),
    m_weight_decay(weight_decay),
    m_momentum(momentum)
    {
       
        if(m_momentum){
            v.reserve(parameters.size());
        }

        if (parameters.size() > 100){
            #pragma omp parallel for 
            for (int64_t i = 0; i < parameters.size() ; i++){
                    v.push_back(Tensor::zeros_like(parameters[i]));
            }
       }else {
            for (int64_t i = 0; i < parameters.size() ; i++){
                    v.push_back(Tensor::zeros_like(parameters[i]));
            }
       }


    }

    void SGD::step() {

        for(int64_t i = 0; i < m_parameters.size(); i++){
            const Tensor& param = m_parameters[i];
            const Tensor& grad = param.grad();
           
            if (!grad.tensor_impl()) continue;

            int64_t size = param.numel();

            if (m_momentum){

                fused_sgd_m_kernel_avx512(
                    param.data_ptr() + param.data_offset(),
                    grad.data_ptr() + param.data_offset(),
                    v[i].data_ptr(),
                    size,
                    m_momentum,
                    m_lr,
                    m_weight_decay
                );

            }else{

                fused_sgd_kernel_avx512(
                    param.data_ptr() + param.data_offset(),
                    grad.data_ptr() + param.data_offset(),
                    size,
                    m_lr,
                    m_weight_decay
                );
                
            }
        }

    }

    std::vector<Tensor>& SGD::stats_v(){
        return v;
    }


}//optimizer

}//nn
}//mt
