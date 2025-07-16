#include <MTensor/nn.hpp>

namespace mt {
namespace nn {

    ConvTranspose1dImpl::ConvTranspose1dImpl(int64_t in_channels, int64_t out_channels, int64_t kernel_size, bool _bias , int64_t stride , int64_t padding_l, int64_t padding_r):
    convTranspose1d({stride}, {padding_l}, {padding_r})
    {
        Shape weights_shape({out_channels, in_channels, kernel_size});

        int64_t fan_in =  in_channels * kernel_size;

        init::kaiming_uniform_ init_weights( fan_in );
    
        weights = paramter(init_weights(weights_shape));

        weights.make_requires_grad();

        if (_bias){
            Shape bias_shape({out_channels});

            float bound = 1.0f / std::sqrt(fan_in);
            
            bias = paramter(Tensor::rand(bias_shape, -bound, bound));
            
            bias.make_requires_grad();
        }
    }

    Tensor ConvTranspose1dImpl::forward(Tensor input) {
        return bias.tensor_impl() ? convTranspose1d.forward({input.tensor_impl(), weights.tensor_impl(), bias.tensor_impl()}) : convTranspose1d.forward({input.tensor_impl(), weights.tensor_impl()});
    }

    MTENSOR_API std::shared_ptr<Module> ConvTranspose1d(int64_t in_channels, int64_t out_channels, int64_t kernel_size, bool _bias , int64_t stride , int64_t padding_l, int64_t padding_r){
        return std::make_shared<ConvTranspose1dImpl>(in_channels, out_channels, kernel_size,  _bias , stride , padding_l, padding_r);
    }
    


    ConvTranspose2dImpl::ConvTranspose2dImpl(int64_t in_channels, int64_t out_channels, const std::vector<int64_t>& kernel_size, bool _bias , const std::vector<int64_t>& stride , const std::vector<int64_t>& padding_l, const std::vector<int64_t>& padding_r):
    convTranspose2d(stride, padding_l, padding_r)
    {
        if ( kernel_size.size() != 2 ){
            throw std::invalid_argument("error : ConvTranspose2d kernel size must be of 2 elements");
        }

        Shape weights_shape({out_channels, in_channels, kernel_size[0], kernel_size[1]});

        int64_t fan_in =  in_channels * kernel_size[0] * kernel_size[1];

        init::kaiming_uniform_ init_weights( fan_in );
    
        weights = paramter(init_weights(weights_shape));

        weights.make_requires_grad();

        if (_bias){
            Shape bias_shape({out_channels});

            float bound = 1.0f / std::sqrt(fan_in);
            
            bias = paramter(Tensor::rand(bias_shape, -bound, bound));
            
            bias.make_requires_grad();
        }
    }

    Tensor ConvTranspose2dImpl::forward(Tensor input) {
        return bias.tensor_impl() ? convTranspose2d.forward({input.tensor_impl(), weights.tensor_impl(), bias.tensor_impl()}) : convTranspose2d.forward({input.tensor_impl(), weights.tensor_impl()});
    }

    MTENSOR_API std::shared_ptr<Module> ConvTranspose2d(int64_t in_channels, int64_t out_channels, const std::vector<int64_t>& kernel_size, bool _bias , const std::vector<int64_t>& stride , const std::vector<int64_t>& padding_l, const std::vector<int64_t>& padding_r){
        return std::make_shared<ConvTranspose2dImpl>(in_channels, out_channels, kernel_size,  _bias , stride , padding_l, padding_r);
    }


    ConvTranspose3dImpl::ConvTranspose3dImpl(int64_t in_channels, int64_t out_channels, const std::vector<int64_t>& kernel_size, bool _bias , const std::vector<int64_t>& stride , const std::vector<int64_t>& padding_l, const std::vector<int64_t>& padding_r):
    convTranspose3d(stride, padding_l, padding_r)
    {
        if ( kernel_size.size() != 3 ){
            throw std::invalid_argument("error : ConvTranspose3d kernel size must be of 3 elements");
        }

        Shape weights_shape({out_channels, in_channels, kernel_size[0], kernel_size[1], kernel_size[2]});

        int64_t fan_in =  in_channels * kernel_size[0] * kernel_size[1] * kernel_size[2];

        init::kaiming_uniform_ init_weights( fan_in );
    
        weights = paramter(init_weights(weights_shape));

        weights.make_requires_grad();

        if (_bias){
            Shape bias_shape({out_channels});

            float bound = 1.0f / std::sqrt(fan_in);
            
            bias = paramter(Tensor::rand(bias_shape, -bound, bound));
            
            bias.make_requires_grad();
        }
    }

    Tensor ConvTranspose3dImpl::forward(Tensor input) {
        return bias.tensor_impl() ? convTranspose3d.forward({input.tensor_impl(), weights.tensor_impl(), bias.tensor_impl()}) : convTranspose3d.forward({input.tensor_impl(), weights.tensor_impl()});
    }

    MTENSOR_API std::shared_ptr<Module> ConvTranspose3d(int64_t in_channels, int64_t out_channels, const std::vector<int64_t>& kernel_size, bool _bias , const std::vector<int64_t>& stride , const std::vector<int64_t>& padding_l, const std::vector<int64_t>& padding_r){
        return std::make_shared<ConvTranspose3dImpl>(in_channels, out_channels, kernel_size,  _bias , stride , padding_l, padding_r);
    }

}//nn
}//mt
