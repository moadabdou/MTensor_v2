#include <gtest/gtest.h>
#include "MTensor/tensorImpl.hpp"
#include <iostream>
#include <windows.h>
#include <utility>


TEST(reduction_ops , all){
    

    mt::ops::Slice  s({{0,mt::EOD}, {0,mt::EOD}, {0,mt::EOD},{1,2}});
    mt::ops::Expand e({3,2,4,4});  
    mt::ops::Mean mean(1);
    mt::ops::Max_reduction max_reduction(2);
    mt::ops::Min_reduction min_reduction(2);
    mt::ops::Mul_reduction mul_reduction({0,2});
    mt::ops::Sum sum({0,2});
    mt::ops::Norm_lp_sum norm_lp_sum({0,2});
    mt::ops::Norm_lp_power_p_sum norm_lp_power_p_sum({0,2}, 3.0f);
        

    std::shared_ptr<mt::TensorImpl> t1 = mt::TensorImpl::randn({3,2,4,2}, 0.0f, 2.0f, 1,true);
    std::shared_ptr<mt::TensorImpl> out_grad = mt::TensorImpl::randn({3,2,1,2}, 0.0f, 2.0f, 1);
    std::shared_ptr<mt::TensorImpl> t2 = e.forward({s.forward({t1})});
    
    std::cout << t1;
    std::cout << out_grad;
    
    auto out = max_reduction.forward({t1});
    std::cout << out;
    out->grad_fn()->backward(out_grad);
    out->grad_fn()->backward(out_grad);
    std::cout << t1->get_grad();
    // std::cout << mean.forward({t2});
    // std::cout << min_reduction.forward({t2});
    // std::cout << mul_reduction.forward({t2});
    // std::cout << sum.forward({t2});
    // std::cout << norm_lp_sum.forward({t2});
    // std::cout << _msize(norm_lp_power_p_sum.forward({t2})->data_ptr().get());

}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::GTEST_FLAG(filter) = "reduction_ops.all";
    return RUN_ALL_TESTS();
}