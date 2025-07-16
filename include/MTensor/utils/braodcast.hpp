#ifndef BROADCAST_H
#define BROADCAST_H 
#include <utility>
#include <memory>
#include <config/mtensor_export.hpp>

namespace mt{

class TensorImpl;

namespace utils{

MTENSOR_API std::pair<std::shared_ptr<TensorImpl>, std::shared_ptr<TensorImpl>> broadcast(const std::pair<std::shared_ptr<TensorImpl>, std::shared_ptr<TensorImpl>>& in_tensors );
MTENSOR_API std::pair<std::shared_ptr<TensorImpl>, std::shared_ptr<TensorImpl>> broadcast_matmul(const std::pair<std::shared_ptr<TensorImpl>, std::shared_ptr<TensorImpl>>& in_tensors );
MTENSOR_API std::shared_ptr<TensorImpl> broadcast_to_shape(const std::shared_ptr<TensorImpl>& in_tensor, const std::vector<int64_t>& to_shape);


} //utils

} //mt 


#endif //BROADCAST_H