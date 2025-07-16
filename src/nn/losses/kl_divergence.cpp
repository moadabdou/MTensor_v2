#include <MTensor/nn.hpp>

namespace mt {
namespace nn {

    KLDivLoss::KLDivLoss(){}

    Tensor KLDivLoss::forward(Tensor output, Tensor target){
        auto loss = ( target * ( target.log() - output) ).sum();
        return loss *  ( 1.0f / static_cast<float>(output.shape()[0]) );
    }

}//nn
}//mt
