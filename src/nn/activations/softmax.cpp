#include <MTensor/nn.hpp>

namespace mt {
namespace nn {

    SoftmaxImpl::SoftmaxImpl(Dim dim): softmax(dim){}

    Tensor SoftmaxImpl::forward(Tensor input){
        return softmax.forward({input.tensor_impl()});
    }

    std::shared_ptr<Module> Softmax(Dim dim){
        return std::make_shared<SoftmaxImpl>(dim);
    }

    LogSoftmaxImpl::LogSoftmaxImpl(Dim dim): logsoftmax(dim){}

    Tensor LogSoftmaxImpl::forward(Tensor input){
        return logsoftmax.forward({input.tensor_impl()});
    }
    std::shared_ptr<Module> LogSoftmax(Dim dim){
        return std::make_shared<LogSoftmaxImpl>(dim);
    }


}//nn
}//mt