#include <gtest/gtest.h>
#include "MTensor/tensorImpl.hpp"
#include <dnnl.hpp>
#include <iostream>
#include <utility>
#include <variant>


TEST(conv , 1d){
    mt::ops::Expand e({2,2,2});
    mt::ops::Squeeze sq(0);
    mt::ops::Slice  s({{0,1}});  
    mt::ops::Contiguous c;
    mt::ops::Cat ct(0);
    mt::ops::View v({8});

    mt::ops::Conv1d conv1d(
        {1},
        {0},
        {0}
    );

    std::shared_ptr<mt::TensorImpl> t1 = s.forward({ mt::TensorImpl::randn({5,6,4}, 0.0f,1.0f, 42, true)}); //B,
    std::shared_ptr<mt::TensorImpl> t2 = mt::TensorImpl::randn({64,4,1024}, 0.0f,1.0f, 42, true);
    std::shared_ptr<mt::TensorImpl> t3 = mt::TensorImpl::randn({64,4,16}, 0.0f,1.0f, 42, true);
    std::shared_ptr<mt::TensorImpl> t4 = mt::TensorImpl::ones({64},true);
    conv1d.forward({t2,t3,t4});
}

TEST(conv , 2d){
    mt::ops::Expand e({2,2,2});
    mt::ops::Squeeze sq(0);
    mt::ops::Slice  s({{0,1}});  
    mt::ops::Contiguous c;
    mt::ops::Cat ct(0);
    mt::ops::View v({8});

    mt::ops::Conv2d conv2d(
        {1,1},
        {0,0},
        {0,0}
    );

    std::shared_ptr<mt::TensorImpl> t1 = s.forward({ mt::TensorImpl::randn({5,6,4}, 0.0f,1.0f, 42, true)}); //B,
    std::shared_ptr<mt::TensorImpl> t2 = mt::TensorImpl::randn({1,3,6,6}, 0.0f,1.0f, 42, true);
    std::shared_ptr<mt::TensorImpl> t3 = mt::TensorImpl::randn({2,3,2,2}, 0.0f,1.0f, 42, true);
    std::shared_ptr<mt::TensorImpl> t4 = mt::TensorImpl::ones({2},true);
    std::cout << t2 << t3;
    std::cout << conv2d.forward({t2,t3,t4});
}


TEST(conv , 3d){
    mt::ops::Expand e({2,2,2});
    mt::ops::Squeeze sq(0);
    mt::ops::Slice  s({{0,1}});  
    mt::ops::Contiguous c;
    mt::ops::Cat ct(0);
    mt::ops::View v({8});

    mt::ops::Conv3d conv3d(
        {1,1,1},
        {1,1,1},
        {1,1,1}
    );

    std::shared_ptr<mt::TensorImpl> t1 = s.forward({ mt::TensorImpl::randn({5,6,4}, 0.0f,1.0f, 42, true)}); //B,
    std::shared_ptr<mt::TensorImpl> t2 = mt::TensorImpl::randn({1,3,4,6,6}, 0.0f,1.0f, 42, true);
    std::shared_ptr<mt::TensorImpl> t3 = mt::TensorImpl::randn({2,3,3,3,3}, 0.0f,1.0f, 42, true);
    std::shared_ptr<mt::TensorImpl> t4 = mt::TensorImpl::ones({2},true);
    std::cout << t2 << t3;
    std::cout << conv3d.forward({t2,t3,t4});
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::GTEST_FLAG(filter) = "conv.3d";
    return RUN_ALL_TESTS();
}