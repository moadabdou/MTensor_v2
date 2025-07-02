#include <iostream>
#include <utility>
#include <gtest/gtest.h>
#include <MTensor/tensorImpl.hpp>


TEST(broadcasting , broadcast){
    mt::ops::Expand e({2,2,2});
    mt::ops::Slice  s({{3,5}});  
    mt::ops::Contiguous c;
    mt::ops::Cat ct(0);
    mt::ops::View v({8});

    std::shared_ptr<mt::TensorImpl> t1 = mt::TensorImpl::rand({2,2,2}, 2,10, true);
    std::shared_ptr<mt::TensorImpl> t2 = mt::TensorImpl::rand({2,2,2}, 2,10, true);
    std::shared_ptr<mt::TensorImpl> t3 = mt::TensorImpl::rand({2,2,1}, 2,10, true); 

    auto [t1_b, t2_b] = mt::utils::broadcast({t1, t2});

    std::cout << t1_b << std::endl;
    std::cout << t2_b << std::endl;

}

TEST(broadcasting , broadcast_matmul){


    std::shared_ptr<mt::TensorImpl> t1 = mt::TensorImpl::rand({3,2,2,4}, 2,10, true);
    std::shared_ptr<mt::TensorImpl> t2 = mt::TensorImpl::rand({4,1}, 2,10, true);

    auto [t1_b, t2_b] = mt::utils::broadcast_matmul({t1, t2});

    std::cout << t1_b << std::endl;
    mt::utils::print_vector(t1_b->shape());
    std::cout << t2_b << std::endl;
    mt::utils::print_vector(t2_b->shape());
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::GTEST_FLAG(filter) = "broadcasting.broadcast_matmul";
    return RUN_ALL_TESTS();
}