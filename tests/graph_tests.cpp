#include <gtest/gtest.h>
#include "MTensor/tensorImpl.hpp"
#include <MTensor/graph.hpp>
#include <iostream>
#include <utility>


TEST(graph , grad){
    
    
    mt::ops::Expand expand({2,2,2});
    
    mt::ops::Slice  slice({{0,mt::EOD}, {0,mt::EOD},{0,200}});  
    
    mt::ops::Contiguous contiguous;
    mt::ops::Cat cat(0);
    mt::ops::View view({8});
    mt::ops::Mean mean({2,3});
    mt::ops::Pow  square(2);
    mt::ops::Sub sub;
    mt::ops::Div div;
    mt::ops::Mul mul;
    mt::ops::Add add;
    mt::ops::Eq eq;
    

    mt::ops::Exp exp;
    mt::ops::Abs abs;
    mt::ops::Clip clip(.4,.7);
    mt::ops::Log log;
    mt::ops::Sqrt sqrt;
    mt::ops::Sigmoid sg;
    mt::ops::Pow pow(3);
    mt::ops::Relu relu(.3);
    mt::ops::Tanh tanh;
    mt::ops::LayerNormalization ln;
    

    std::shared_ptr<mt::TensorImpl> t1 = mt::TensorImpl::rand({200,200,200}, -2,2, -1);
    std::shared_ptr<mt::TensorImpl> t2 = mt::TensorImpl::rand({200,1,200}, -2,2, -1, true);
    std::shared_ptr<mt::TensorImpl> t3 = mt::TensorImpl::rand({200,200,1}, -2,2, -1, true);
    std::shared_ptr<mt::TensorImpl> t4 = mt::TensorImpl::rand({200,200,400}, -2,2, -1, true);
    std::shared_ptr<mt::TensorImpl> t5 = mt::TensorImpl::rand({200,200,400}, -2,2, -1, true);
    std::shared_ptr<mt::TensorImpl> t12 = mt::TensorImpl::rand({200}, 0,1, -1, true);
    std::shared_ptr<mt::TensorImpl> t13 = mt::TensorImpl::rand({200}, -1,1, -1, true);


    std::cout << "init done";

    auto t6 = add.forward({t4,t5}); //2,2,4

    auto t7 = mul.forward({t1,t2}); //2,2,2

    auto t8 = div.forward({t7,t3}); //2,2,2

    auto t9 = slice.forward({t6});

    auto t10 = tanh.forward({t9});

    auto t11 = sub.forward({t10, t8});

    auto t14 = ln.forward({t11, t12, t13});

    std::cout << "calc done";


    mt::graph::GradGraph graph(t14);

    graph.export_to("graph.html");

   

}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::GTEST_FLAG(filter) = "graph.grad";
    return RUN_ALL_TESTS();
}