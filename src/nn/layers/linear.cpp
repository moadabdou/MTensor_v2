#include <MTensor/nn.hpp>

namespace mt {
namespace nn {

    LinearImpl::LinearImpl(int64_t in_features, int64_t out_features, bool _bias, int64_t seed ){
        
        Shape weights_shape({in_features, out_features});

        init::kaiming_uniform_ init_weights(in_features, "leaky_relu", std::sqrt(5.0f));
    
        m_weights = paramter(init_weights(weights_shape, seed));

        m_weights.make_requires_grad();

        if (_bias){
            Shape bias_shape({out_features});

            float bound = 1.0f / std::sqrt(in_features);
            
            m_bias = paramter(Tensor::rand(bias_shape, -bound, bound, seed));
            
            m_bias.make_requires_grad();
        }

    } 
    
    Tensor LinearImpl::forward(Tensor input){
        return m_bias.tensor_impl() ? matmul.forward({input.tensor_impl(), m_weights.tensor_impl(), m_bias.tensor_impl()}) : matmul.forward({input.tensor_impl(), m_weights.tensor_impl()});
    }

    Tensor LinearImpl::weights() const{
        return m_weights;
    }
    Tensor LinearImpl::bias() const{
        return m_bias;
    }

    MTENSOR_API std::shared_ptr<Module> Linear(int64_t in_features, int64_t out_features, bool _bias, int64_t seed ){
        return std::make_shared<LinearImpl>(in_features, out_features, _bias, seed );
    }

}//nn
}//mt
