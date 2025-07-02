#include <gtest/gtest.h>
#include "MTensor/tensorImpl.hpp"
#include <dnnl.hpp>
#include <iostream>
#include <utility>
#include <variant>


TEST(linear_ops , matmul){
    mt::ops::Expand e({2,2,2});
    mt::ops::Squeeze sq(0);
    mt::ops::Slice  s({{0,1}});  
    mt::ops::Contiguous c;
    mt::ops::Cat ct(0);
    mt::ops::View v({8});

    mt::ops::Matmul matmul;

    std::shared_ptr<mt::TensorImpl> t1 = sq.forward({s.forward({ mt::TensorImpl::randn({5,6,3}, 0.0f,1.0f, 42, true)})}); //B,
    std::shared_ptr<mt::TensorImpl> t2 = mt::TensorImpl::randn({3,2}, 0.0f,1.0f, 42, true);
    std::shared_ptr<mt::TensorImpl> t3 = mt::TensorImpl::randn({2}, 0.0f,1.0f, 42, true);

    std::cout << t1 << t2 << t3;
    std::cout  << matmul.forward({t1, t2, t3});
    

}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::GTEST_FLAG(filter) = "linear_ops.matmul";
    return RUN_ALL_TESTS();
}