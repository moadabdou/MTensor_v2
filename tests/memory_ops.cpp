#include <gtest/gtest.h>
#include "MTensor/tensorImpl.hpp"
#include <iostream>
#include <utility>
#include <variant>

TEST(memory_ops , clone){

    std::shared_ptr<mt::TensorImpl> t1 = mt::TensorImpl::rand({3,1,2,2}, 2,10, true);
    mt::ops::Expand slc({3,4,2,2});
    mt::ops::Clone clone_op;

    t1 = slc.forward({t1});
    std::shared_ptr<mt::TensorImpl> t_out =  clone_op.forward({t1});
    std::cout << t_out << std::endl;
    std::cout << t1;
    std::cout << t_out->data_offset() << std::endl;
    // Print strides and shape of t1
    std::cout << "t1 shape: ";
    for (auto s : t1->shape()) std::cout << s << " ";
    std::cout << "\nt1 strides: ";
    for (auto s : t1->stride()) std::cout << s << " ";
    std::cout << std::endl;

    // Print strides and shape of t_out
    std::cout << "t_out shape: ";
    for (auto s : t_out->shape()) std::cout << s << " ";
    std::cout << "\nt_out strides: ";
    for (auto s : t_out->stride()) std::cout << s << " ";
    std::cout << std::endl;

    std::cout << t1->data_offset();

}

TEST(memory_ops , contiguous){

    std::shared_ptr<mt::TensorImpl> t1 = mt::TensorImpl::rand({30,50,2,2}, 2,10, true);
    mt::ops::Slice slc({{1,2},{2,4}});
    mt::ops::Contiguous contiguous_op;

    t1 = slc.forward({t1});
    std::shared_ptr<mt::TensorImpl> t_out =  contiguous_op.forward({t1});
    std::cout << t_out << std::endl;
    std::cout << t1;
    std::cout << t_out->data_offset() << std::endl;
    std::cout << t1->data_offset();

}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::GTEST_FLAG(filter) = "memory_ops.clone";
    return RUN_ALL_TESTS();
}