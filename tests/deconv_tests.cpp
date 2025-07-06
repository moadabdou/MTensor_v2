#include <gtest/gtest.h>
#include "MTensor/tensorImpl.hpp"
#include <dnnl.hpp>
#include <iostream>
#include <utility>
#include <variant>


TEST(deconv , 1d){
    mt::ops::Expand e({2,2,2});
    mt::ops::Squeeze sq(0);
    mt::ops::Slice  s({{0,1}});  
    mt::ops::Contiguous c;
    mt::ops::Cat ct(0);
    mt::ops::View v({8});

    mt::ops::Deconv1d deconv1d(
        {2},
        {0},
        {0}
    );

    std::shared_ptr<mt::TensorImpl> t2 = mt::TensorImpl::randn({1,3,4}, 0.0f,1.0f, 42, true);
    std::shared_ptr<mt::TensorImpl> t3 = mt::TensorImpl::randn({2,3,2}, 0.0f,1.0f, 42, true);
    std::shared_ptr<mt::TensorImpl> t4 = mt::TensorImpl::ones({2},true);
    std::shared_ptr<mt::TensorImpl> t5 = mt::TensorImpl::randn({1,2,8}, 0.0f,1.0f, 43);

    std::cout << t2 << t3 << t4 << t5;

    auto out = deconv1d.forward({t2,t3,t4});

    std::cout << out;

    out->grad_fn()->backward(t5);
    out->grad_fn()->backward(t5);

    std::cout << t2->get_grad() << t3->get_grad()  << t4->get_grad() ;
}

TEST(deconv , 2d){
    mt::ops::Expand e({2,2,2});
    mt::ops::Squeeze sq(0);
    mt::ops::Slice  s({{0,1}});  
    mt::ops::Contiguous c;
    mt::ops::Cat ct(0);
    mt::ops::View v({8});

    mt::ops::Deconv2d deconv2d(
        {2,2},
        {0,0},
        {0,0}
    );

    std::shared_ptr<mt::TensorImpl> t2 = mt::TensorImpl::randn({3,2,5,5}, 0.0f,1.0f, 42, true);
    std::shared_ptr<mt::TensorImpl> t3 = mt::TensorImpl::ones({3,4,3,3}, true);
    std::shared_ptr<mt::TensorImpl> t4 = mt::TensorImpl::ones({64},true); // 9  
    std::cout << t2 << t3 ;
    std::cout << deconv2d.forward({t2,t3});
}


TEST(deconv , 3d){
    mt::ops::Expand e({2,2,2});
    mt::ops::Squeeze sq(0);
    mt::ops::Slice  s({{0,1}});  
    mt::ops::Contiguous c;
    mt::ops::Cat ct(0);
    mt::ops::View v({8});

    mt::ops::Deconv3d deconv3d(
        {2,2,2},
        {0,0,0},
        {0,0,0}
    );

    std::shared_ptr<mt::TensorImpl> t1 = s.forward({ mt::TensorImpl::randn({5,1,5}, 0.0f,1.0f, 42, true)}); //B,
    std::shared_ptr<mt::TensorImpl> t2 = mt::TensorImpl::randn({3,2,5,5,5}, 0.0f,1.0f, 42, true);
    std::shared_ptr<mt::TensorImpl> t3 = mt::TensorImpl::randn({3,2,3,3,3}, 0.0f,1.0f, 42,true);
    std::shared_ptr<mt::TensorImpl> t4 = mt::TensorImpl::randn({3}, 0.0f,1.0f, 42,true); // 9  
    std::cout << t2 << t3 << t4 ;
    std::cout << deconv3d.forward({t2,t3,t4});
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::GTEST_FLAG(filter) = "deconv.1d";
    return RUN_ALL_TESTS();
}