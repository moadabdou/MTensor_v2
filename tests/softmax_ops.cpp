#include <gtest/gtest.h>
#include "MTensor/tensorImpl.hpp"
#include <dnnl.hpp>
#include <iostream>
#include <utility>
#include <variant>


TEST(softmax , softmax){
    mt::ops::Expand e({2,2,2});
    mt::ops::Squeeze sq(0);
    mt::ops::Squeeze sq_(1);
    mt::ops::Slice  s({{0,1}});  
    mt::ops::Contiguous c;
    mt::ops::Cat ct(0);
    mt::ops::Sum sum({1});
    mt::ops::View v({8});

    mt::ops::Softmax softmax(1);

    std::shared_ptr<mt::TensorImpl> t1 = sq.forward({s.forward({ mt::TensorImpl::randn({5,6,3}, 0.0f,1.0f, 42, true)})}); //B,
    std::shared_ptr<mt::TensorImpl> t2 = mt::TensorImpl::randn({3,2}, 0.0f,1.0f, 42, true);
    std::shared_ptr<mt::TensorImpl> t3 = mt::TensorImpl::randn({2}, 0.0f,1.0f, 42, true);

    std::cout << t2;
    std::cout  << softmax.forward({t2});

}


TEST(softmax , softmax_log){
    mt::ops::Expand e({2,2,2});
    mt::ops::Squeeze sq(0);
    mt::ops::Squeeze sq_(1);
    mt::ops::Slice  s({{0,1}});  
    mt::ops::Contiguous c;
    mt::ops::Cat ct(0);
    mt::ops::Sum sum({1});
    mt::ops::View v({8});

    mt::ops::SoftmaxLog softmax_log(1);

    std::shared_ptr<mt::TensorImpl> t1 = sq.forward({s.forward({ mt::TensorImpl::randn({5,6,3}, 0.0f,1.0f, 42, true)})}); //B,
    std::shared_ptr<mt::TensorImpl> t2 = mt::TensorImpl::randn({3,2}, 0.0f,1.0f, 42, true);
    std::shared_ptr<mt::TensorImpl> t3 = mt::TensorImpl::randn({2}, 0.0f,1.0f, 42, true);

    std::cout << t2;
    std::cout  << softmax_log.forward({t2});

}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::GTEST_FLAG(filter) = "softmax.softmax_log";
    return RUN_ALL_TESTS();
}