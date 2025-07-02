#include <gtest/gtest.h>
#include "MTensor/tensorImpl.hpp"


TEST(init_tensor, init_tensor_Implt){

    std::shared_ptr<mt::TensorImpl> t1 = mt::TensorImpl::rand({2,2,3,5}, 2,10);
    std::cout << t1;

}





