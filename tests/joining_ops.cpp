#include <gtest/gtest.h>
#include "MTensor/tensorImpl.hpp"
#include <dnnl.hpp>
#include <iostream>
#include <utility>
#include <variant>


TEST(joining_ops , cat){
    mt::ops::Expand e({2,2,2});
    mt::ops::Slice  s({{3,5}});  
    mt::ops::Contiguous c;
    mt::ops::Cat ct(0);
    mt::ops::View v({8});

    std::shared_ptr<mt::TensorImpl> t1 = mt::TensorImpl::rand({2,2,2}, 2,10, true);
    std::shared_ptr<mt::TensorImpl> t2 = mt::TensorImpl::rand({1,2,2}, 2,10, true);
    std::shared_ptr<mt::TensorImpl> t3 = mt::TensorImpl::rand({2,2,1}, 2,10, true); 



    t3 = e.forward({t3});
    std::shared_ptr<mt::TensorImpl> t4 = ct.forward({t1,t2,t3});
    std::shared_ptr<mt::TensorImpl> t5 = s.forward({t4});
    std::shared_ptr<mt::TensorImpl> t6 = c.forward({t5});
    // Print the stride vector
    std::cout << t5 << t6;
    std::cout << v.forward({t6});

}

TEST(joining_ops , stack){
    mt::ops::Slice e({{20,23},{15,18}});
    mt::ops::Slice  s({{3,5}});  
    mt::ops::Contiguous c;
    mt::ops::Stack st(2);
    mt::ops::View v({8});

    std::shared_ptr<mt::TensorImpl> t1 = mt::TensorImpl::rand({3,3}, 2,10, true);
    std::shared_ptr<mt::TensorImpl> t2 = mt::TensorImpl::rand({3,3}, 2,10, true);
    std::shared_ptr<mt::TensorImpl> t3 = mt::TensorImpl::rand({30,20}, 2,10, true); 


    t3 = e.forward({t3});
    std::shared_ptr<mt::TensorImpl> t4 = st.forward({t1,t2,t3});
    std::cout << t1;
    std::cout << t2;
    std::cout << t3;
    std::cout << t4;

}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::GTEST_FLAG(filter) = "joining_ops.stack";
    return RUN_ALL_TESTS();
}