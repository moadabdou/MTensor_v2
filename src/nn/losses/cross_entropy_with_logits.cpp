#include <MTensor/nn.hpp>

namespace mt {
namespace nn {

    BCEWithLogitsLoss::BCEWithLogitsLoss(std::string reduction): m_reduction(reduction){}

    Tensor BCEWithLogitsLoss::forward(Tensor output, Tensor target){
        Tensor max_part = output.relu();

        Tensor x_mul_y = output * target;

        Tensor log_term = ((output.abs() * -1.f ).exp() +  1).log();

        Tensor loss = max_part - x_mul_y + log_term;
        loss = loss.flat();

        if (m_reduction == "mean") {
            return loss.mean(0);
        } else if (m_reduction == "sum"){
            return loss.sum({0});
        } else {
            return loss;
        }  
    }
}//nn
}//mt
