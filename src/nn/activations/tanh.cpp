#include <MTensor/nn.hpp>

namespace mt {
namespace nn {

    TanhImpl::TanhImpl(){}

    Tensor TanhImpl::forward(Tensor input){
        return tanh.forward({input.tensor_impl()});
    }

    std::shared_ptr<Module> Tanh(){
        return std::make_shared<TanhImpl>();
    }

}//nn
}//mt