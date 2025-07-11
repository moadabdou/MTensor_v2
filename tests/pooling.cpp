#include <gtest/gtest.h>
#include "MTensor/tensorImpl.hpp"
#include <dnnl.hpp>
#include <iostream>
#include <utility>
#include <variant>


TEST(pooling , max_1d){
    mt::ops::Expand e({2,2,2});
    mt::ops::Squeeze sq(0);
    mt::ops::Slice  s({{0,1}});  
    mt::ops::Contiguous c;
    mt::ops::Cat ct(0);
    mt::ops::View v({8});

    mt::ops::MaxPooling1d max1d(
        {4},
        {4},
        {0},
        {0}
    );

    std::shared_ptr<mt::TensorImpl> t1 = s.forward({ mt::TensorImpl::randn({5,6,4}, 0.0f,1.0f, 42, true)}); //B,
    std::shared_ptr<mt::TensorImpl> t2 = mt::TensorImpl::randn({4,10,20}, 0.0f,1.0f, 42, true);
    std::shared_ptr<mt::TensorImpl> t4 = mt::TensorImpl::randn({4,10,5}, 0.0f,1.0f, 41);
    std::shared_ptr<mt::TensorImpl> t3 = mt::TensorImpl::randn({2}, 0.0f,1.0f, 42, true);

    std::cout << t2;
    auto out =  max1d.forward({t2});
    std::cout << t4;
    std::cout  << out;
    out->grad_fn()->backward(t4);
    std::cout << t2->get_grad();

    

    
}

TEST(pooling , max_2d){
    mt::ops::Expand e({2,2,2});
    mt::ops::Squeeze sq(0);
    mt::ops::Slice  s({{0,1}});  
    mt::ops::Contiguous c;
    mt::ops::Cat ct(0);
    mt::ops::View v({8});

    mt::ops::MaxPooling2d max2d(
        {2,2},
        {2,2},
        {0,0},
        {0,0}
    );

    std::shared_ptr<mt::TensorImpl> t1 = s.forward({ mt::TensorImpl::randn({5,6,4}, 0.0f,1.0f, 42, true)}); //B,
    std::shared_ptr<mt::TensorImpl> t2 = mt::TensorImpl::randn({4,3,8,8}, 0.0f,1.0f, 42, true);
    std::shared_ptr<mt::TensorImpl> t4 = mt::TensorImpl::randn({4,3,4,4}, 0.0f,1.0f, 41);
    std::shared_ptr<mt::TensorImpl> t3 = mt::TensorImpl::randn({2}, 0.0f,1.0f, 42, true);

    std::cout << t2;
    auto out =  max2d.forward({t2});
    std::cout << t4;
    std::cout  << out;
    out->grad_fn()->backward(t4);
    std::cout << t2->get_grad();

    
}


TEST(pooling , max_3d){
    mt::ops::Expand e({2,2,2});
    mt::ops::Squeeze sq(0);
    mt::ops::Slice  s({{0,1}});  
    mt::ops::Contiguous c;
    mt::ops::Cat ct(0);
    mt::ops::View v({8});

    mt::ops::MaxPooling3d max3d(
        {2,2,2},
        {2,2,2},
        {0,0,0},
        {0,0,0}
    );

    std::shared_ptr<mt::TensorImpl> t1 = s.forward({ mt::TensorImpl::randn({5,6,4}, 0.0f,1.0f, 42, true)}); //B,
    std::shared_ptr<mt::TensorImpl> t2 = mt::TensorImpl::randn({4,3,4,4,4}, 0.0f,1.0f, 42, true);
    std::shared_ptr<mt::TensorImpl> t3 = mt::TensorImpl::randn({2}, 0.0f,1.0f, 42, true);

    std::cout << t2;
    std::cout  << max3d.forward({t2});
}


TEST(pooling , avg_1d){

    mt::ops::AvgPooling1d avg1d(
        {2},
        {2},
        {0},
        {0}
    );

    std::shared_ptr<mt::TensorImpl> t2 = mt::TensorImpl::randn({4,3,8}, 0.0f,1.0f, 42, true);
    std::shared_ptr<mt::TensorImpl> t4 = mt::TensorImpl::randn({4,3,4}, 0.0f,1.0f, 41);
    std::shared_ptr<mt::TensorImpl> t3 = mt::TensorImpl::randn({2}, 0.0f,1.0f, 42, true);

    std::cout << t2;
    auto out =  avg1d.forward({t2});
    std::cout << t4;
    std::cout  << out;
    out->grad_fn()->backward(t4);
    std::cout << t2->get_grad();

}


TEST(pooling , avg_2d){
    mt::ops::Expand e({2,2,2});
    mt::ops::Squeeze sq(0);
    mt::ops::Slice  s({{0,1}});  
    mt::ops::Contiguous c;
    mt::ops::Cat ct(0);
    mt::ops::View v({8});

    mt::ops::AvgPooling2d avg2d(
        {2,2},
        {2,2},
        {0,0},
        {0,0}
    );

    std::shared_ptr<mt::TensorImpl> t1 = s.forward({ mt::TensorImpl::randn({5,6,4}, 0.0f,1.0f, 42, true)}); //B,
    std::shared_ptr<mt::TensorImpl> t2 = mt::TensorImpl::randn({4,3,8,8}, 0.0f,1.0f, 42, true);
    std::shared_ptr<mt::TensorImpl> t4 = mt::TensorImpl::randn({4,3,4,4}, 0.0f,1.0f, 41);
    std::shared_ptr<mt::TensorImpl> t3 = mt::TensorImpl::randn({2}, 0.0f,1.0f, 42, true);

    std::cout << t2;
    auto out =  avg2d.forward({t2});
    std::cout << t4;
    std::cout  << out;
    out->grad_fn()->backward(t4);
    std::cout << t2->get_grad();  
}

TEST(pooling , avg_3d){
    mt::ops::Expand e({2,2,2});
    mt::ops::Squeeze sq(0);
    mt::ops::Slice  s({{0,1}});  
    mt::ops::Contiguous c;
    mt::ops::Cat ct(0);
    mt::ops::View v({8});

    mt::ops::AvgPooling3d avg3d(
        {2,2,2},
        {2,2,2},
        {0,0,0},
        {0,0,0},
        true
    );

    std::shared_ptr<mt::TensorImpl> t1 = s.forward({ mt::TensorImpl::randn({5,6,4}, 0.0f,1.0f, 42, true)}); //B,
    std::shared_ptr<mt::TensorImpl> t2 = mt::TensorImpl::randn({1,1,2,4,4}, 0.0f,1.0f, 42, true);
    std::shared_ptr<mt::TensorImpl> t3 = mt::TensorImpl::randn({2}, 0.0f,1.0f, 42, true);

    std::cout << t2;
    std::cout  << avg3d.forward({t2});
    
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::GTEST_FLAG(filter) = "pooling.avg_2d";
    return RUN_ALL_TESTS();
}