#include <gtest/gtest.h>
#include "MTensor/tensorImpl.hpp"
#include <dnnl.hpp>
#include <iostream>
#include <utility>
#include <variant>


TEST(normalization , batch){
    mt::ops::Expand e({2,2,2});
    mt::ops::Squeeze sq(0);
    mt::ops::Slice  s({{0,1}});  
    mt::ops::Contiguous c;
    mt::ops::Cat ct(0);
    mt::ops::View v({8});
    mt::ops::Mean mean({0,2,3});
    mt::ops::Pow  square(2);
    mt::ops::Sqrt sqrt;
    mt::ops::Sub sub;

    std::shared_ptr<mt::TensorImpl> src = mt::TensorImpl::randn({1000,5,4,4}, 2.0f,1.0f, 42, true);
    std::shared_ptr<mt::TensorImpl> src_1 = mt::TensorImpl::randn({1000,5,4,4}, 2.0f,1.0f, 42, true);
    std::shared_ptr<mt::TensorImpl> src_2 = mt::TensorImpl::randn({1,5,4,4}, 2.0f,1.0f, 42, true);
    std::shared_ptr<mt::TensorImpl> scale = mt::TensorImpl::rand({5}, 3,5,true);
    std::shared_ptr<mt::TensorImpl> shift = mt::TensorImpl::randn({5},true);
    std::shared_ptr<mt::TensorImpl> running_mean = mt::TensorImpl::zeros({5});
    std::shared_ptr<mt::TensorImpl> running_var = mt::TensorImpl::zeros({5});   

    mt::ops::BatchNormalization normalization(true, running_mean, running_var);
    mt::ops::BatchNormalization normalization_inf(false, running_mean, running_var);

    auto out = normalization.forward({src, scale, shift});

    auto _mean = mean.forward({out});
    auto _mean_2 = mean.forward({square.forward({out})});

    std::cout <<scale ;
    std::cout << sqrt.forward({ sub.forward({_mean_2, square.forward({ _mean}) }) });
    
}

TEST(normalization , layer){
    mt::ops::Expand e({2,2,2});
    mt::ops::Squeeze sq(0);
    mt::ops::Slice  s({{0,1}});  
    mt::ops::Contiguous c;
    mt::ops::Cat ct(0);
    mt::ops::View v({8});
    mt::ops::Mean mean({2});
    mt::ops::Pow  square(2);
    mt::ops::Sqrt sqrt;
    mt::ops::Sub sub;
    mt::ops::Div div;
    mt::ops::Mul mul;
    mt::ops::Add add;

    std::shared_ptr<mt::TensorImpl> src = mt::TensorImpl::randn({2,3,4}, 2.0f,1.0f, 42, true);
    std::shared_ptr<mt::TensorImpl> scale = mt::TensorImpl::randn({4},true);
    std::shared_ptr<mt::TensorImpl> shift = mt::TensorImpl::randn({4},true); 



    mt::ops::LayerNormalization normalization;

    auto out = normalization.forward({src, scale, shift});

    auto _mean = mean.forward({src});
    auto _mean_2 = mean.forward({square.forward({src})});
    auto _std = sqrt.forward({ sub.forward({_mean_2, square.forward({ _mean}) }) });

    std::cout << _mean;
    std::cout << _std;
    std::cout << add.forward({ mul.forward({scale, div.forward({  sub.forward({ src , _mean })  , _std})}) , shift});
    std::cout << out;
    
}


TEST(normalization , group){
    mt::ops::Expand e({2,2,2});
    mt::ops::Squeeze sq(0);
    mt::ops::Slice  s({{0,1}});  
    mt::ops::Contiguous c;
    mt::ops::Cat ct(0);
    mt::ops::View v({8});
    mt::ops::Mean mean({2,3});
    mt::ops::Pow  square(2);
    mt::ops::Sqrt sqrt;
    mt::ops::Sub sub;
    mt::ops::Div div;
    mt::ops::Mul mul;
    mt::ops::Add add;
    mt::ops::Eq eq;
    mt::ops::View view({2,2,2,4});
    mt::ops::View view_r({2,4,4});

    std::shared_ptr<mt::TensorImpl> src = mt::TensorImpl::randn({2,4,4}, 2.0f,1.0f, 42, true);
    std::shared_ptr<mt::TensorImpl> scale = mt::TensorImpl::ones({4},true);
    std::shared_ptr<mt::TensorImpl> shift = mt::TensorImpl::zeros({4},true); 



    mt::ops::GroupNormalization normalization(2);
    auto out = normalization.forward({src, scale, shift});

    auto src_g = view.forward({src});
    auto _mean = mean.forward({ src_g});
    auto _mean_2 =   mean.forward({square.forward({  src_g })});
    auto _std =  sqrt.forward({ sub.forward({ _mean_2, square.forward({_mean})  }) });
    auto man_out = view_r.forward({div.forward({sub.forward({ src_g , _mean }), _std})});

    std::cout << src;
    std::cout << _mean;
    std::cout << _std;
    std::cout <<  man_out;
    std::cout << out;
    std::cout << sub.forward({man_out, out});
    
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::GTEST_FLAG(filter) = "normalization.group";
    return RUN_ALL_TESTS();
}