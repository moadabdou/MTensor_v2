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
    std::shared_ptr<mt::TensorImpl> t2 = mt::TensorImpl::randn({4,3,2}, 0.0f,1.0f, 42, true);
    std::shared_ptr<mt::TensorImpl> t3 = mt::TensorImpl::randn({4,3,2}, 0.0f,1.0f, 40);

    std::cout << t2;
    auto out = softmax.forward({t2});
    out->grad_fn()->backward(t3);
    std::cout << out;
    std::cout << t3;
    std::cout << t2->get_grad();

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
    std::shared_ptr<mt::TensorImpl> t2 = mt::TensorImpl::randn({4,3,2}, 0.0f,1.0f, 42, true);
    std::shared_ptr<mt::TensorImpl> t3 = mt::TensorImpl::randn({4,3,2}, 0.0f,1.0f, 40);

    std::cout << t2;
    auto out = softmax_log.forward({t2});
    out->grad_fn()->backward(t3);
    std::cout << out;
    std::cout << t3;
    std::cout << t2->get_grad();

}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::GTEST_FLAG(filter) = "softmax.softmax_log";
    return RUN_ALL_TESTS();
}