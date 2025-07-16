#include <MTensor/nn.hpp>

namespace mt {
namespace nn {

    CrossEntropyLoss::CrossEntropyLoss(){}

    Tensor CrossEntropyLoss::forward(Tensor output, Tensor target){
        auto loss = ( output.logsoftmax(1) * target ).sum() ;
        return loss * (-1.0f / static_cast<float>(output.shape()[0]));
    }

}//nn
}//mt
