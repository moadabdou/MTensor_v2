#include <gtest/gtest.h>
#include "MTensor/tensorImpl.hpp"
#include <iostream>
#include <utility>
#include <variant>

TEST(view_grad , view){

    std::shared_ptr<mt::TensorImpl> t1 = mt::TensorImpl::rand({2,3,4,5}, 2,10,42, true);
    std::shared_ptr<mt::TensorImpl> t2 = mt::TensorImpl::rand({2,12,5}, 2,10,42);
    
    mt::ops::View view({2,12,5});

    std::shared_ptr<mt::TensorImpl> t_out =  view.forward({t1});
    std::cout << t1 << t2;
    std::cout << t_out;
    t_out->grad_fn()->backward(t2);
    t_out->grad_fn()->backward(t2);
    std::cout << t1->get_grad();
    std::cout << t1->get_grad()->data_ptr() <<  "   " << t2->data_ptr();
}

TEST(view_grad , transpose){

    std::shared_ptr<mt::TensorImpl> t1 = mt::TensorImpl::rand({2,3,4,5}, 2,10,42, true);
    std::shared_ptr<mt::TensorImpl> t2 = mt::TensorImpl::rand({2,4,3,5}, 2,10,42);
    
    mt::ops::Transpose tr2d_op(1,2);

    std::shared_ptr<mt::TensorImpl> t_out =  tr2d_op.forward({t1});
    std::cout << t1 << t2;
    std::cout << t_out;
    t_out->grad_fn()->backward(t2);
    std::cout << t1->get_grad()->data_ptr() <<  "   " << t2->data_ptr();

}


TEST(view_grad , permute){

    std::shared_ptr<mt::TensorImpl> t1 = mt::TensorImpl::rand({2,4,3}, 2,10, 42,true);
    std::shared_ptr<mt::TensorImpl> t2 = mt::TensorImpl::rand({3,2,4}, 2,10, 42);
    
    mt::ops::Permute pmt({2,0,1});

    std::shared_ptr<mt::TensorImpl> t_out =  pmt.forward({t1});
    std::cout << t1;
    std::cout << t2;
    std::cout << t_out;
    t_out->grad_fn()->backward(t2);
    t_out->grad_fn()->backward(t2);
    std::cout << t1->get_grad();
    std::cout << t1->get_grad()->data_ptr() <<  "   " << t2->data_ptr();

}

TEST(view_grad , squeeze_and_unsqueeze){

    std::shared_ptr<mt::TensorImpl> t1 = mt::TensorImpl::rand({2,1,3,4}, 2,10,42, true);
    std::shared_ptr<mt::TensorImpl> t2 = mt::TensorImpl::rand({2,3,4}, 2,10,42);
    
    mt::ops::Squeeze sqz(1);

    std::shared_ptr<mt::TensorImpl> t_out =  sqz.forward({t1});
    std::cout << t1;
    std::cout << t2;
    std::cout << t_out;
    t_out->grad_fn()->backward(t2);
    t_out->grad_fn()->backward(t2);
    std::cout << t1->get_grad();
    std::cout << t1->get_grad()->data_ptr() <<  "   " << t2->data_ptr();

}



TEST(view_grad , narrow){

    std::shared_ptr<mt::TensorImpl> t1 = mt::TensorImpl::rand({2,4,4}, 2,10, 42 , true);
    std::shared_ptr<mt::TensorImpl> t2 = mt::TensorImpl::ones({1,4,4});
    std::shared_ptr<mt::TensorImpl> t3 = mt::TensorImpl::rand({1,4,4});
    
    mt::ops::Narrow nrw(0,0,1); //1,4,4
    mt::ops::Narrow nrw1(0,1,1); //1,4,4

    std::shared_ptr<mt::TensorImpl> t_out =  nrw.forward({t1});
    std::shared_ptr<mt::TensorImpl> t_out_1 =  nrw1.forward({t1});
    std::cout << t1;
    std::cout << t_out;
    std::cout << t_out_1;
    std::cout << t3;
    mt::utils::print_vector(t3->stride());
    t_out->grad_fn()->backward(t2);
    t_out->grad_fn()->backward(t2);
    t_out_1->grad_fn()->backward(t3);
    t_out_1->grad_fn()->backward(t3);
    std::cout << t1->get_grad();
    
}

TEST(view_grad , slice){

    std::shared_ptr<mt::TensorImpl> t1 = mt::TensorImpl::rand({2,4,3}, 2,10,42,true);
    std::shared_ptr<mt::TensorImpl> t2 = mt::TensorImpl::rand({1,2,1}, 2,10,42,true);
    
    mt::ops::Slice slc({{1,2},{0,2},{2,3}});

    std::shared_ptr<mt::TensorImpl> t_out =  slc.forward({t1});
    std::cout << t1;
    std::cout << t2;
    std::cout << t_out;
    t_out->grad_fn()->backward(t2);
    t_out->grad_fn()->backward(t2);
    std::cout << t1->get_grad();
    std::cout << t1->get_grad()->data_ptr() <<  "   " << t2->data_ptr();

}

TEST(view_grad , expand){

    std::shared_ptr<mt::TensorImpl> t1 = mt::TensorImpl::rand({2,5,1},0,1,42 ,true);
    std::shared_ptr<mt::TensorImpl> t2 = mt::TensorImpl::ones({3,2,5,4});

    mt::ops::Expand expd({3,2,5,4});
    std::shared_ptr<mt::TensorImpl> t_out =  expd.forward({t1});

    std::cout << t1;
    std::cout << t_out;
    t_out->grad_fn()->backward(t2);
    std::cout << t1->get_grad();
    t_out->grad_fn()->backward(t2);
    std::cout << t1->get_grad();

}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::GTEST_FLAG(filter) = "view_grad.view";
    return RUN_ALL_TESTS();
}


