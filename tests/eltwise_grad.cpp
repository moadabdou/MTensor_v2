#include <gtest/gtest.h>
#include "MTensor/tensorImpl.hpp"
#include <iostream>
#include <utility>


TEST(eltwise_grad , all){
    
    
    mt::ops::Expand e({2,2,2});
    
    mt::ops::Slice  s({{0,mt::EOD}, {0,mt::EOD},{0,1}});  
    
    //mt::ops::Contiguous c;
    mt::ops::Cat ct(0);
    mt::ops::View v({8});
    
    mt::ops::Linear linear(2,2);
    mt::ops::Exp exp;
    mt::ops::Abs abs;
    mt::ops::Clip clip(.4,.7);
    mt::ops::Log log;
    mt::ops::Sqrt sqrt;
    mt::ops::Sigmoid sg;
    mt::ops::Pow pow(3);
    mt::ops::Relu relu(.3);
    mt::ops::Tanh tanh;
    

    std::shared_ptr<mt::TensorImpl> t1 = mt::TensorImpl::rand({2,4,4}, 0,1,42,true);
    std::shared_ptr<mt::TensorImpl> t2 = mt::TensorImpl::ones({2,4,4});

    std::cout << t1;
    auto out = log.forward({t1});
    std::cout << out;
    out->grad_fn()->backward(t2);
    out->grad_fn()->backward(t2);
    std::cout << t1->get_grad();

}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::GTEST_FLAG(filter) = "eltwise_grad.all";
    return RUN_ALL_TESTS();
}