#include <MTensor/nn.hpp>


void fill_mask(float* d_ptr, int64_t N, float p){
    srand(time(0));
    for(int64_t i = 0; i < N; i++){
        d_ptr[i] = std::rand() / (float)RAND_MAX > p ?  1.0f/(1.0f - p) : 0.0f;
    }
}

namespace mt {
namespace nn {

    Dropout1dImpl::Dropout1dImpl(float p): m_p(p){}

    Tensor Dropout1dImpl::forward(Tensor input){
        auto input_shape = input.shape();
        if (input_shape.size() != 2 && input_shape.size() != 3 ){
            throw std::invalid_argument("Dropout1d() invalid input, must 2d or 3d tensor");
        }

        Shape mask_shape =  input_shape.size() == 2 ? 
            Shape{input_shape[0],input_shape[1]} : 
            Shape{input_shape[0],input_shape[1],1};
        
        auto mask = std::make_shared<TensorImpl>(mask_shape);

        fill_mask(mask->data_ptr().get() + mask->data_offset(), input_shape[0]*input_shape[1], m_p);

        return mul.forward({mask, input.tensor_impl()});
    }


    MTENSOR_API std::shared_ptr<Module> Dropout1d(float p){
        return std::make_shared<Dropout1dImpl>(p);
    }

    Dropout2dImpl::Dropout2dImpl(float p): m_p(p){}

    Tensor Dropout2dImpl::forward(Tensor input){
        auto input_shape = input.shape();
        if (input_shape.size() != 4 ){
            throw std::invalid_argument("Dropout2d() invalid input, must 4d tensor");
        }

        Shape mask_shape = Shape{input_shape[0],input_shape[1],1,1};
        
        auto mask = std::make_shared<TensorImpl>(mask_shape);

   
        fill_mask(mask->data_ptr().get() + mask->data_offset(), input_shape[0]*input_shape[1], m_p);

        return mul.forward({mask, input.tensor_impl()});
    }

    MTENSOR_API std::shared_ptr<Module> Dropout2d(float p){
        return std::make_shared<Dropout2dImpl>(p);
    }

    Dropout3dImpl::Dropout3dImpl(float p): m_p(p){}

    Tensor Dropout3dImpl::forward(Tensor input){
        auto input_shape = input.shape();
        if (input_shape.size() != 5 ){
            throw std::invalid_argument("Dropout3d() invalid input, must 5d tensor");
        }

        Shape mask_shape = Shape{input_shape[0],input_shape[1],1,1,1};
        
        auto mask = std::make_shared<TensorImpl>(mask_shape);

        fill_mask(mask->data_ptr().get() + mask->data_offset(), input_shape[0]*input_shape[1], m_p);

        return mul.forward({mask, input.tensor_impl()});
    }

    MTENSOR_API std::shared_ptr<Module> Dropout3d(float p){
        return std::make_shared<Dropout3dImpl>(p);
    }

}//nn
}//mt
