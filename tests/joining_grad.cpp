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
    mt::ops::Cat ct(1);
    mt::ops::View v({8});

    std::shared_ptr<mt::TensorImpl> t1 = mt::TensorImpl::rand({3,2,4,3}, 2,10, 100,true);
    std::shared_ptr<mt::TensorImpl> t2 = mt::TensorImpl::rand({3,5,4,3}, 2,10, 42,true);
    std::shared_ptr<mt::TensorImpl> t3 = mt::TensorImpl::rand({3,4,4,3}, 2,10, 66,true);
    std::shared_ptr<mt::TensorImpl> t5 = mt::TensorImpl::rand({3,11,4,3}, 2,10, 22);  

    std::shared_ptr<mt::TensorImpl> t4 = ct.forward({t1,t2,t3});

    t4->grad_fn()->backward(t5);
    std::cout<< t3->get_grad() ;
    mt::utils::print_vector (t4->shape());
    mt::utils::print_vector (t1->get_grad()->shape());
    mt::utils::print_vector (t2->get_grad()->shape());
    mt::utils::print_vector (t3->get_grad()->shape());

}

TEST(joining_ops , stack){
    mt::ops::Expand e({2,2,2});
    mt::ops::Slice  s({{3,5}});  
    mt::ops::Contiguous c;
    mt::ops::Stack st(1);
    mt::ops::View v({8});

    std::shared_ptr<mt::TensorImpl> t1 = mt::TensorImpl::rand({3,2,4}, 2,10, 100,true);
    std::shared_ptr<mt::TensorImpl> t2 = mt::TensorImpl::rand({3,2,4}, 2,10, 42,true);
    std::shared_ptr<mt::TensorImpl> t3 = mt::TensorImpl::rand({3,2,4}, 2,10, 66,true);
    std::shared_ptr<mt::TensorImpl> t5 = mt::TensorImpl::rand({3,3,2,4}, 2,10, 22);  

    std::shared_ptr<mt::TensorImpl> t4 = st.forward({t1,t2,t3});
    t4->grad_fn()->backward(t5);
    std::cout << t1 << t2  << t3 << t5 ;
    std::cout << t1->get_grad() << t2->get_grad()  << t3->get_grad() ;
    std::cout << t1->get_grad()->data_ptr() << t2->get_grad()->data_ptr()  << t3->get_grad()->data_ptr() ;
    mt::utils::print_vector (t4->shape());
    mt::utils::print_vector (t1->get_grad()->shape());
    mt::utils::print_vector (t2->get_grad()->shape());
    mt::utils::print_vector (t3->get_grad()->shape());

}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::GTEST_FLAG(filter) = "joining_ops.stack";
    return RUN_ALL_TESTS();
}