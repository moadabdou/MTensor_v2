#include <MTensor/nn.hpp>

namespace mt {
namespace nn {

    L1Loss::L1Loss(){}

    Tensor L1Loss::forward(Tensor output, Tensor target){
        auto loss =  ( output - target ).abs().sum();
        return loss * ( 1.0f / static_cast<float>(output.shape()[0]) );
    }

}//nn
}//mt
