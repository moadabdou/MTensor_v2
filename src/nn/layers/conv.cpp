#include <MTensor/nn.hpp>

namespace mt {
namespace nn {

    Conv1dImpl::Conv1dImpl(int64_t in_channels, int64_t out_channels, int64_t kernel_size, bool _bias , int64_t stride , int64_t padding_l, int64_t padding_r):
    conv1d({stride}, {padding_l}, {padding_r})
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

    Tensor Conv1dImpl::forward(Tensor input) {
        return bias.tensor_impl() ? conv1d.forward({input.tensor_impl(), weights.tensor_impl(), bias.tensor_impl()}) : conv1d.forward({input.tensor_impl(), weights.tensor_impl()});
    }

    MTENSOR_API std::shared_ptr<Module> Conv1d(int64_t in_channels, int64_t out_channels, int64_t kernel_size, bool _bias , int64_t stride , int64_t padding_l, int64_t padding_r){
        return std::make_shared<Conv1dImpl>(in_channels, out_channels, kernel_size,  _bias , stride , padding_l, padding_r);
    }


    Conv2dImpl::Conv2dImpl(int64_t in_channels, int64_t out_channels, const std::vector<int64_t>& kernel_size, bool _bias , const std::vector<int64_t>& stride , const std::vector<int64_t>& padding_l, const std::vector<int64_t>& padding_r):
    conv2d(stride, padding_l, padding_r)
    {
        if ( kernel_size.size() != 2 ){
            throw std::invalid_argument("error : Conv2d kernel size must be of 2 elements");
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

    Tensor Conv2dImpl::forward(Tensor input) {
        return bias.tensor_impl() ? conv2d.forward({input.tensor_impl(), weights.tensor_impl(), bias.tensor_impl()}) : conv2d.forward({input.tensor_impl(), weights.tensor_impl()});
    }

    MTENSOR_API std::shared_ptr<Module> Conv2d(int64_t in_channels, int64_t out_channels, const std::vector<int64_t>& kernel_size, bool _bias , const std::vector<int64_t>& stride , const std::vector<int64_t>& padding_l, const std::vector<int64_t>& padding_r){
        return std::make_shared<Conv2dImpl>(in_channels, out_channels, kernel_size,  _bias , stride , padding_l, padding_r);
    }


    Conv3dImpl::Conv3dImpl(int64_t in_channels, int64_t out_channels, const std::vector<int64_t>& kernel_size, bool _bias , const std::vector<int64_t>& stride , const std::vector<int64_t>& padding_l, const std::vector<int64_t>& padding_r):
    conv3d(stride, padding_l, padding_r)
    {
        if ( kernel_size.size() != 3 ){
            throw std::invalid_argument("error : Conv3d kernel size must be of 3 elements");
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

    Tensor Conv3dImpl::forward(Tensor input) {
        return bias.tensor_impl() ? conv3d.forward({input.tensor_impl(), weights.tensor_impl(), bias.tensor_impl()}) : conv3d.forward({input.tensor_impl(), weights.tensor_impl()});
    }

    MTENSOR_API std::shared_ptr<Module> Conv3d(int64_t in_channels, int64_t out_channels, const std::vector<int64_t>& kernel_size, bool _bias , const std::vector<int64_t>& stride , const std::vector<int64_t>& padding_l, const std::vector<int64_t>& padding_r){
        return std::make_shared<Conv3dImpl>(in_channels, out_channels, kernel_size,  _bias , stride , padding_l, padding_r);
    }

}//nn
}//mt
