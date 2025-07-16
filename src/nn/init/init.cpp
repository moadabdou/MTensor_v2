#include <MTensor/nn.hpp>

namespace mt {
namespace init{

    float calculate_gain(std::string nonlinearity, float a){
        if (nonlinearity == "linear") {
            return 1.0f;
        } else if (nonlinearity == "conv") {
            return 1.0f;
        } else if (nonlinearity == "sigmoid") {
            return 1.0f;
        } else if (nonlinearity == "tanh") {
            return 5.0f / 3;
        } else if (nonlinearity == "relu") {
            return std::sqrt(2.0f);
        } else if (nonlinearity == "leaky_relu") {
            return std::sqrt(2.0f / (1.0f + a * a));
        } else {
            return 1.0f;
        }
    }



}//init
}//mt
