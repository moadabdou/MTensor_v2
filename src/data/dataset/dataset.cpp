
#include <MTensor/data.hpp>
#include <MTensor/tensor.hpp>
#include <omp.h>

namespace mt{
namespace data{


    
    MTENSOR_API int64_t compare_output_target(const Tensor& output, const Tensor& target){

        if ( output.shape() != target.shape() || output.shape().size() != 2){
            throw std::invalid_argument (" compare_output_target(): output and traget must be both 2d Tensors ");
        }

        auto max_output = output.max(1).indices;
        auto max_target = target.max(1).indices;


        int64_t res = 0;
        int32_t size = max_output.size();

        // #pragma omp parallel for reduction( + : res ) usually the batch size is relatively small so no need for parallelism
        for (int64_t i = 0; i < size ;  i ++){
            if ( max_output[i][1] == max_target[i][1] ) res += 1; 
        }

        return res;
    }



}//data

} //mt