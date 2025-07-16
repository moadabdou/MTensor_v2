#include <MTensor/nn.hpp>

namespace mt {
namespace nn {

    MSELoss::MSELoss(){}

    Tensor MSELoss::forward(Tensor output, Tensor target){
        auto loss = ( output - target ).pow(2).sum();
        return loss *  ( 1.0f / static_cast<float>(output.shape()[0]));
    }

}//nn
}//mt
