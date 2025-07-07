#include <gtest/gtest.h>
#include "MTensor/tensorImpl.hpp"
#include <dnnl.hpp>
#include <iostream>
#include <utility>
#include <variant>


TEST(normalization , batch){

    std::shared_ptr<mt::TensorImpl> src = mt::TensorImpl::randn({2,5,4,4}, 2.0f,1.0f, 42, true);
    std::shared_ptr<mt::TensorImpl> scale = mt::TensorImpl::randn({5}, 2.0f,1.0f, 42, true);
    std::shared_ptr<mt::TensorImpl> shift = mt::TensorImpl::randn({5}, 2.0f,1.0f, 42, true);
    std::shared_ptr<mt::TensorImpl> grad = mt::TensorImpl::randn({2,5,4,4}, 2.0f,1.0f, 43);
    std::shared_ptr<mt::TensorImpl> running_mean = mt::TensorImpl::zeros({5});
    std::shared_ptr<mt::TensorImpl> running_var = mt::TensorImpl::zeros({5});   

    mt::ops::BatchNormalization normalization(true, running_mean, running_var);
    mt::ops::BatchNormalization normalization_inf(false, running_mean, running_var);

    auto out = normalization.forward({src, scale, shift});
    out->grad_fn()->backward(grad);
    out->grad_fn()->backward(grad);
    std::cout << out ;
    std::cout << src->get_grad() << scale->get_grad() << shift->get_grad();

    
}

TEST(normalization , layer){
   
    std::shared_ptr<mt::TensorImpl> src = mt::TensorImpl::randn({2,5,4,4}, 2.0f,1.0f, 42, true);
    std::shared_ptr<mt::TensorImpl> scale = mt::TensorImpl::randn({4}, 2.0f,1.0f, 42, true);
    std::shared_ptr<mt::TensorImpl> shift = mt::TensorImpl::randn({4}, 2.0f,1.0f, 42, true);
    std::shared_ptr<mt::TensorImpl> grad = mt::TensorImpl::randn({2,5,4,4}, 2.0f,1.0f, 43);  

    mt::ops::LayerNormalization normalization;

    auto out = normalization.forward({src, scale, shift});
    out->grad_fn()->backward(grad);
    out->grad_fn()->backward(grad);
    std::cout << out ;
    std::cout << src->get_grad() << scale->get_grad() << shift->get_grad();
    
}


TEST(normalization , group){
    
    std::shared_ptr<mt::TensorImpl> src = mt::TensorImpl::randn({2,6,4}, 2.0f,1.0f, 42, true);
    std::shared_ptr<mt::TensorImpl> scale = mt::TensorImpl::randn({6}, 2.0f,1.0f, 42, true);
    std::shared_ptr<mt::TensorImpl> shift = mt::TensorImpl::randn({6}, 2.0f,1.0f, 42, true);
    std::shared_ptr<mt::TensorImpl> grad = mt::TensorImpl::randn({2,6,4}, 2.0f,1.0f, 43);  

    mt::ops::GroupNormalization normalization(3);
    mt::ops::LayerNormalization layernormalization;
    std::cout << grad;
    auto out = normalization.forward({src, scale, shift});
    out->grad_fn()->backward(grad);
    out->grad_fn()->backward(grad);

    std::cout  << src->get_grad() << scale->get_grad() << shift->get_grad();
    
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::GTEST_FLAG(filter) = "normalization.group";
    return RUN_ALL_TESTS();
}