#include <MTensor/nn.hpp>

namespace mt {
namespace nn {

    BatchNormImpl::BatchNormImpl(int64_t num_features, bool& training,float momentum):
    m_training(training),
    m_momentum(momentum)
    {
        running_mean = training ? auxiliary(Tensor::zeros({num_features})) : Tensor();
        running_var  = training ? auxiliary(Tensor::zeros({num_features})) : Tensor();
        gamma = paramter(Tensor::ones({num_features}, training));
        beta  = paramter(Tensor::zeros({num_features}, training));
    }

    Tensor BatchNormImpl::forward(Tensor input){
        auto bn = ops::BatchNormalization(m_training, running_mean.tensor_impl(), running_var.tensor_impl(), m_momentum);
        return bn.forward({input.tensor_impl(), gamma.tensor_impl(), beta.tensor_impl()});
    }

    MTENSOR_API std::shared_ptr<Module> BatchNorm(int64_t num_features, bool& training,float momentum){
        return std::make_shared<BatchNormImpl>(num_features, training, momentum);
    }






    LayerNormImpl::LayerNormImpl(int64_t num_features){
        gamma = paramter(Tensor::ones({num_features}, true));
        beta  = paramter(Tensor::zeros({num_features}, true));
    }

    Tensor LayerNormImpl::forward(Tensor input){
        return ln.forward({input.tensor_impl(), gamma.tensor_impl(), beta.tensor_impl()});
    }

    MTENSOR_API std::shared_ptr<Module> LayerNorm(int64_t num_features){
        return std::make_shared<LayerNormImpl>(num_features);
    }






    
    RMSNormImpl::RMSNormImpl(int64_t num_features){
        gamma = paramter(Tensor::ones({num_features}, true));
    }

    Tensor RMSNormImpl::forward(Tensor input){
        return rmsn.forward({input.tensor_impl(), gamma.tensor_impl()});
    }

    MTENSOR_API std::shared_ptr<Module> RMSNorm(int64_t num_features){
        return std::make_shared<RMSNormImpl>(num_features);
    }




    GroupNormImpl::GroupNormImpl(int64_t groups, int64_t num_channels):
    gn(groups){
        gamma = paramter(Tensor::ones({num_channels}, true));
        beta  = paramter(Tensor::zeros({num_channels}, true));
    }

    Tensor GroupNormImpl::forward(Tensor input){
        return gn.forward({input.tensor_impl(), gamma.tensor_impl(), beta.tensor_impl()});
    }

    MTENSOR_API std::shared_ptr<Module> GroupNorm(int64_t groups, int64_t num_channels){
        return std::make_shared<GroupNormImpl>(groups,num_channels);
    }

}//nn
}//mt
