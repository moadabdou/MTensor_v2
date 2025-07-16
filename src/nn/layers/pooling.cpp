#include <MTensor/nn.hpp>

namespace mt {
namespace nn {

    MaxPooling1dImpl::MaxPooling1dImpl(int64_t pooling_size, int64_t stride , int64_t padding_l, int64_t padding_r):
    maxpooling1d({pooling_size}, {stride}, {padding_l}, {padding_r})
    {}

    Tensor MaxPooling1dImpl::forward(Tensor input) {
        return maxpooling1d.forward({input.tensor_impl()});
    }

    MTENSOR_API std::shared_ptr<Module> MaxPooling1d(int64_t pooling_size, int64_t stride , int64_t padding_l, int64_t padding_r){
        return std::make_shared<MaxPooling1dImpl>(pooling_size, stride , padding_l, padding_r);
    }


    

    MaxPooling2dImpl::MaxPooling2dImpl(const std::vector<int64_t>& pooling_size, const std::vector<int64_t>& stride , const std::vector<int64_t>& padding_l, const std::vector<int64_t>& padding_r):
    maxpooling2d(pooling_size, stride, padding_l, padding_r)
    {}

    Tensor MaxPooling2dImpl::forward(Tensor input) {
        return maxpooling2d.forward({input.tensor_impl()});
    }

    MTENSOR_API std::shared_ptr<Module> MaxPooling2d(const std::vector<int64_t>& pooling_size, const std::vector<int64_t>& stride , const std::vector<int64_t>& padding_l, const std::vector<int64_t>& padding_r){
        return std::make_shared<MaxPooling2dImpl>(pooling_size, stride , padding_l, padding_r);
    }



    MaxPooling3dImpl::MaxPooling3dImpl(const std::vector<int64_t>& pooling_size, const std::vector<int64_t>& stride , const std::vector<int64_t>& padding_l, const std::vector<int64_t>& padding_r):
    maxpooling3d(pooling_size, stride, padding_l, padding_r)
    {}

    Tensor MaxPooling3dImpl::forward(Tensor input) {
        return maxpooling3d.forward({input.tensor_impl()});
    }

    MTENSOR_API std::shared_ptr<Module> MaxPooling3d(const std::vector<int64_t>& pooling_size, const std::vector<int64_t>& stride , const std::vector<int64_t>& padding_l, const std::vector<int64_t>& padding_r){
        return std::make_shared<MaxPooling3dImpl>(pooling_size, stride , padding_l, padding_r);
    }



    AvgPooling1dImpl::AvgPooling1dImpl(int64_t pooling_size, int64_t stride , int64_t padding_l, int64_t padding_r, bool include_padding):
    avgpooling1d({pooling_size}, {stride}, {padding_l}, {padding_r}, include_padding)
    {}

    Tensor AvgPooling1dImpl::forward(Tensor input) {
        return avgpooling1d.forward({input.tensor_impl()});
    }

    MTENSOR_API std::shared_ptr<Module> AvgPooling1d(int64_t pooling_size, int64_t stride , int64_t padding_l, int64_t padding_r, bool include_padding){
        return std::make_shared<AvgPooling1dImpl>(pooling_size, stride , padding_l, padding_r, include_padding);
    }



    AvgPooling2dImpl::AvgPooling2dImpl(const std::vector<int64_t>& pooling_size, const std::vector<int64_t>& stride , const std::vector<int64_t>& padding_l, const std::vector<int64_t>& padding_r, bool include_padding ):
    avgpooling2d(pooling_size, stride, padding_l, padding_r, include_padding)
    {}


    Tensor AvgPooling2dImpl::forward(Tensor input) {
        return avgpooling2d.forward({input.tensor_impl()});
    }

    MTENSOR_API std::shared_ptr<Module> AvgPooling2d(const std::vector<int64_t>& pooling_size, const std::vector<int64_t>& stride , const std::vector<int64_t>& padding_l, const std::vector<int64_t>& padding_r, bool include_padding ){
        return std::make_shared<AvgPooling2dImpl>(pooling_size, stride , padding_l, padding_r, include_padding );
    }



    AvgPooling3dImpl::AvgPooling3dImpl(const std::vector<int64_t>& pooling_size, const std::vector<int64_t>& stride , const std::vector<int64_t>& padding_l, const std::vector<int64_t>& padding_r, bool include_padding):
    avgpooling3d(pooling_size, stride, padding_l, padding_r, include_padding)
    {}


    Tensor AvgPooling3dImpl::forward(Tensor input) {
        return avgpooling3d.forward({input.tensor_impl()});
    }

    MTENSOR_API std::shared_ptr<Module> AvgPooling3d(const std::vector<int64_t>& pooling_size, const std::vector<int64_t>& stride , const std::vector<int64_t>& padding_l, const std::vector<int64_t>& padding_r, bool include_padding ){
        return std::make_shared<AvgPooling3dImpl>(pooling_size, stride , padding_l, padding_r, include_padding );
    }

}//nn
}//mt
