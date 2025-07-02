#include <gtest/gtest.h>
#include "MTensor/tensorImpl.hpp"
#include <iostream>
#include <utility>


TEST(binary_ops , all){
    

    mt::ops::Slice  s({{0,mt::EOD}, {0,mt::EOD}, {0,mt::EOD},{1,2}});
    mt::ops::Expand e({3,2,4,4});  
    mt::ops::Add add;
    mt::ops::Sub sub;
    mt::ops::Mul mul;
    mt::ops::Div div;
    mt::ops::Max max;
    mt::ops::Min min;

    mt::ops::Gt gt;
    mt::ops::Ge ge;
    mt::ops::Le le;
    mt::ops::Lt lt;

    mt::ops::Ne ne;
    mt::ops::Eq eq;


    std::shared_ptr<mt::TensorImpl> t1 = mt::TensorImpl::ones({2,4,4}, true);
    std::shared_ptr<mt::TensorImpl> t2 = e.forward({s.forward({mt::TensorImpl::zeros({3,2,1,2}, true)})});
    
    std::cout << t1;
    std::cout << t2;
    std::cout << ne.forward({t1, t2});
    std::cout << eq.forward({t1, t2});

}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::GTEST_FLAG(filter) = "binary_ops.all";
    return RUN_ALL_TESTS();
}