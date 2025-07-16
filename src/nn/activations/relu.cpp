#include <MTensor/nn.hpp>

namespace mt {
namespace nn {

    ReluImpl::ReluImpl(float a): relu(a){}

    Tensor ReluImpl::forward(Tensor input){
        return relu.forward({input.tensor_impl()});
    }

    std::shared_ptr<Module> Relu(float a){
        return std::make_shared<ReluImpl>(a);
    }


}//nn
}//mt
