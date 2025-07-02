#include <gtest/gtest.h>
#include "MTensor/tensorImpl.hpp"
#include <iostream>
#include <utility>
#include <variant>

TEST(view_ops , view){

    std::shared_ptr<mt::TensorImpl> t1 = mt::TensorImpl::rand({3,5}, 2,10);
    
    mt::ops::View view_op({5,3});

    std::shared_ptr<mt::TensorImpl> t_out =  view_op.forward({t1});

    std::cout << t_out;
    std::cout << t1;

}

TEST(view_ops , transpose){

    std::shared_ptr<mt::TensorImpl> t1 = mt::TensorImpl::rand({3,3}, 2,10, true);
    
    mt::ops::Transpose tr2d_op(0,1);

    std::shared_ptr<mt::TensorImpl> t_out =  tr2d_op.forward({t1});
    std::cout << t_out;
    std::cout << t_out->grad_fn()->operands()[0];

}


TEST(view_ops , permute){

    std::shared_ptr<mt::TensorImpl> t1 = mt::TensorImpl::rand({2,3,4}, 2,10, true);
    
    mt::ops::Permute pmt({2,0,1});

    std::shared_ptr<mt::TensorImpl> t_out =  pmt.forward({t1});
    std::cout << t_out;
    std::cout << t_out->grad_fn()->operands()[0];

}

TEST(view_ops , squeeze){

    std::shared_ptr<mt::TensorImpl> t1 = mt::TensorImpl::rand({2,1,3,4}, 2,10, true);
    
    mt::ops::Squeeze sqz(3);

    std::shared_ptr<mt::TensorImpl> t_out =  sqz.forward({t1});
    std::cout << t_out;
    std::cout << t_out->grad_fn()->operands()[0];

}

TEST(view_ops , unsqueeze){

    std::shared_ptr<mt::TensorImpl> t1 = mt::TensorImpl::rand({2,3,4}, 2,10, true);
    
    mt::ops::Unsqueeze usqz(2);

    std::shared_ptr<mt::TensorImpl> t_out =  usqz.forward({t1});
    std::cout << t_out;
    std::cout << t_out->grad_fn()->operands()[0];

}


TEST(view_ops , narrow){

    std::shared_ptr<mt::TensorImpl> t1 = mt::TensorImpl::rand({2,3,4}, 2,10);
    
    mt::ops::Narrow nrw(0,0,1); //1,3,4
    mt::ops::Expand expd({4,3,4});

    std::shared_ptr<mt::TensorImpl> t_out =  expd.forward({nrw.forward({t1})});
    std::cout << t_out;
    
}

TEST(view_ops , slice){

    std::shared_ptr<mt::TensorImpl> t1 = mt::TensorImpl::rand({2,5,4}, 2,10,true);
    
    mt::ops::Slice slc({{0,1},{0,1},{0,1}});

    std::shared_ptr<mt::TensorImpl> t_out =  slc.forward({t1});
    std::cout << t_out;
    std::cout << t_out->grad_fn()->operands()[0];
}

TEST(view_ops , expand){

    std::shared_ptr<mt::TensorImpl> t1 = mt::TensorImpl::rand({2,5,4}, 2,10,true);
    
    mt::ops::Slice s({{0,mt::EOD},{0,mt::EOD},{0,1}});
    mt::ops::Expand expd({3,2,5,4});

    t1 = s.forward({t1});

    std::cout << t1;

    std::shared_ptr<mt::TensorImpl> t_out =  expd.forward({t1});
    std::cout << t_out;
    std::cout << t_out->is_contiguous();
    std::cout << t_out->grad_fn()->operands()[0];
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::GTEST_FLAG(filter) = "view_ops.expand";
    return RUN_ALL_TESTS();
}


