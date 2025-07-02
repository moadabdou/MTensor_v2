#include <gtest/gtest.h>
#include "MTensor/tensorImpl.hpp"
#include <iostream>
#include <utility>


TEST(eltwise_ops , all){
    
    
    mt::ops::Expand e({2,2,2});
    
    mt::ops::Slice  s({{0,mt::EOD}, {0,mt::EOD},{0,1}});  
    
    //mt::ops::Contiguous c;
    mt::ops::Cat ct(0);
    mt::ops::View v({8});
    

    mt::ops::Exp exp;
    mt::ops::Abs abs;
    mt::ops::Clip clip(.4,.7);
    mt::ops::Log log;
    mt::ops::Sqrt sqrt;
    mt::ops::Sigmoid sg;
    mt::ops::Pow pow(3);
    mt::ops::Relu relu(.3);
    mt::ops::Tanh tanh;
    

    std::shared_ptr<mt::TensorImpl> t1 = mt::TensorImpl::rand({2,2,2}, -2,2, true);
   
    
    std::shared_ptr<mt::TensorImpl> t2 = s.forward({t1});

    std::cout << t2;
    std::cout << exp.forward({t2});
    std::cout << log.forward({t2});
    std::cout << sg.forward({t2});
    std::cout << sqrt.forward({t2});
    std::cout << pow.forward({t2});
    std::cout << clip.forward({t2});
    std::cout << relu.forward({t2});
    std::cout << tanh.forward({t2});
    std::cout << abs.forward({t2});

}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::GTEST_FLAG(filter) = "eltwise_ops.all";
    return RUN_ALL_TESTS();
}