#ifndef TENSOR_PRINT_H
#define TENSOR_PRINT_H

#include <stdint.h>
#include <vector>
#include <memory>
#include "config/mtensor_export.hpp"


namespace mt
{
    class TensorImpl;
namespace utils {
    
    MTENSOR_API void print_tensor(const float* data,const int64_t data_offset, const std::vector<int64_t>& shape, const std::vector<int64_t>& stride, int64_t numel, int level = 0, int64_t offset = 0);

}//utils
} // mt
#endif //TENSOR_PRINT_H