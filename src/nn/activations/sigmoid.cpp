#include <MTensor/nn.hpp>

namespace mt {
namespace nn {

    SigmoidImpl::SigmoidImpl(){}

    Tensor SigmoidImpl::forward(Tensor input){
        return sigmoid.forward({input.tensor_impl()});
    }
    
    std::shared_ptr<Module> Sigmoid(){
        return std::make_shared<SigmoidImpl>();
    }

}//nn
}//mt