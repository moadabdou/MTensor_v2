#ifndef PRINT_VECTOR_H
#define PRINT_VECTOR_H 
#include <iostream>
#include <vector>
#include <config/mtensor_export.hpp>

namespace mt{

class TensorImpl;

namespace utils{

template <class T>
void  print_vector(const std::vector<T>& vec){
    std::cout << "Vector([";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << vec[i];
        if (i != vec.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "])" << std::endl;
}

} //utils

} //mt 


#endif //PRINT_VECTOR_H