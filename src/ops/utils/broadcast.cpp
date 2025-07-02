#include <stdexcept>
#include <MTensor/tensorImpl.hpp>

namespace mt{

namespace utils{

std::pair<std::shared_ptr<TensorImpl>, std::shared_ptr<TensorImpl>> broadcast(const std::pair<std::shared_ptr<TensorImpl>, std::shared_ptr<TensorImpl>>& in_tensors ){

    auto [t1, t2] = in_tensors;

    if (t1->shape() == t2->shape()){ //if same shape no broadcasting is needed
        return in_tensors;
    }

    const auto& max_size_tensor = t1->shape().size() > t2->shape().size() ? t1 :  t2 ;
    const auto& min_size_tensor = t1->shape().size() > t2->shape().size() ? t2 :  t1 ;

    //check if they are broadcastable 
    const auto& max_shape = max_size_tensor->shape();
    const auto& min_shape = min_size_tensor->shape();
    auto max_rank = max_shape.size();
    auto min_rank = min_shape.size();

    for (int64_t i = 0; i < min_rank; ++i) {
        int64_t max_dim = max_shape[max_rank - 1 - i];
        int64_t min_dim = min_shape[min_rank - 1 - i];
        if (min_dim != max_dim && min_dim != 1 && max_dim != 1) { 
            throw std::invalid_argument("Tensors are not broadcastable: dimension mismatch.");
        }
    }

    //calculate the broadcasting shape
    std::vector<int64_t> broadcast_shape(max_rank);

    for (int64_t i = 0; i < max_rank; ++i) {
        int64_t max_dim = max_shape[max_rank - 1 - i];
        int64_t min_dim = (i < min_rank) ? min_shape[min_rank - 1 - i] : 1;
        broadcast_shape[max_rank - 1 - i] = std::max(max_dim, min_dim);
    }

    mt::ops::Expand expand(broadcast_shape);

    if (t1->shape() != broadcast_shape){
        t1 = expand.forward({t1});
    }

    if (t2->shape() != broadcast_shape){
        t2 = expand.forward({t2});
    }
    
    return {t1,t2};
}


std::pair<std::shared_ptr<TensorImpl>, std::shared_ptr<TensorImpl>> broadcast_matmul(const std::pair<std::shared_ptr<TensorImpl>, std::shared_ptr<TensorImpl>>& in_tensors ){

    auto [t1, t2] = in_tensors;

    const auto& t1_shape = t1->shape();
    const auto& t2_shape = t2->shape();

    if (t1_shape.size() < 2 || t2_shape.size() < 2){
        throw std::invalid_argument(" matmul broadcasting is allowed only for tensors of dim N where N >= 2 ");
    }    

    if (t1_shape[t1_shape.size()-1] != t2_shape[t2_shape.size() - 2]){
        throw std::invalid_argument(" matmul broadcasting is allowed only for tensors that have valid format for matmul (aka, input_0_shape[-1] = input_1_shape[-2]) ");
    }  

    if (t1_shape.size() == 2 &&  t2_shape.size() == 2){ 
        return in_tensors;
    }

    const auto& max_size_tensor = t1_shape.size() > t2_shape.size() ? t1 :  t2 ;
    const auto& min_size_tensor = t1_shape.size() > t2_shape.size() ? t2 :  t1 ;

    //check if they are broadcastable 
    const auto& max_shape = max_size_tensor->shape();
    const auto& min_shape = min_size_tensor->shape();
    auto max_rank = max_shape.size() - 2; 
    auto min_rank = min_shape.size() - 2; // 3

    for (int64_t i = 0; i < min_rank; ++i) {
        int64_t max_dim = max_shape[max_rank - 1 - i];
        int64_t min_dim = min_shape[min_rank - 1 - i];
        if (min_dim != max_dim && min_dim != 1 && max_dim != 1) { 
            throw std::invalid_argument("Tensors are not broadcastable: dimension mismatch.");
        }
    }

    //calculate the broadcasting shape
    std::vector<int64_t> broadcast_shape(max_rank + 2);

    for (int64_t i = 0; i < max_rank; ++i) {
        int64_t max_dim = max_shape[max_rank - 1 - i];
        int64_t min_dim = (i < min_rank) ? min_shape[min_rank - 1 - i] : 1;
        broadcast_shape[max_rank - 1 - i] = std::max(max_dim, min_dim);
    }

    if (max_size_tensor.get() == t1.get()){

        broadcast_shape[max_rank + 1] = max_shape[max_rank + 1];
        broadcast_shape[max_rank] = max_shape[max_rank];

        if ( max_shape !=  broadcast_shape){
            mt::ops::Expand expd_max({broadcast_shape});
            t1 = expd_max.forward({t1}); 
        }

        broadcast_shape[max_rank + 1] = min_shape[min_rank + 1];
        broadcast_shape[max_rank] = min_shape[min_rank];

        if ( min_shape !=  broadcast_shape){
            mt::ops::Expand expd_min({broadcast_shape});
            t2 = expd_min.forward({t2}); 
        }

    }else {

        broadcast_shape[max_rank + 1] = max_shape[max_rank + 1];
        broadcast_shape[max_rank] = max_shape[max_rank];

        if ( max_shape !=  broadcast_shape){
            mt::ops::Expand expd_max({broadcast_shape});
            t2 = expd_max.forward({t2}); 
        }

        broadcast_shape[max_rank + 1] = min_shape[min_rank + 1];
        broadcast_shape[max_rank] = min_shape[min_rank];

        if ( max_shape !=  broadcast_shape){
            mt::ops::Expand expd_min({broadcast_shape});
            t1 = expd_min.forward({t1}); 
        } 

    }

    return {t1,t2};
}
   

std::shared_ptr<TensorImpl> broadcast_to_shape(const std::shared_ptr<TensorImpl>& in_tensor, const std::vector<int64_t>& to_shape){

    const auto& in_tensor_shape = in_tensor->shape();
    int64_t to_shape_rank =  static_cast<int64_t>(to_shape.size());
    int64_t in_tensor_rank = static_cast<int64_t>(in_tensor_shape.size());

    if (in_tensor_shape == to_shape){ //if same shape no broadcasting is needed
        return in_tensor;
    }

    if (to_shape_rank < in_tensor_rank){
        throw std::invalid_argument(" can't broadcast to to_shape that is sampler than in_tensor.shape()! ");
    }

    for (int64_t i = 0; i < in_tensor_rank; ++i) {
        int64_t max_dim = to_shape[to_shape_rank - 1 - i];
        int64_t min_dim = in_tensor_shape[in_tensor_rank - 1 - i];
        if (min_dim != max_dim && min_dim != 1) {  //(a,a) (1,a)
            throw std::invalid_argument("Tensor is not broadcastable to to_shape : dimension mismatch.");
        }
    }

    //calculate the broadcasting shape
    std::vector<int64_t> broadcast_shape(to_shape_rank);

    for (int64_t i = 0; i < to_shape_rank; ++i) {
        int64_t max_dim = to_shape[to_shape_rank - 1 - i];
        int64_t min_dim = (i < in_tensor_rank) ? in_tensor->shape()[in_tensor_rank - 1 - i] : 1;
        broadcast_shape[to_shape_rank - 1 - i] = std::max(max_dim, min_dim);
    }

    mt::ops::Expand expand(broadcast_shape);
    
    return  expand.forward({in_tensor});
}


}//utils

} //mt 


