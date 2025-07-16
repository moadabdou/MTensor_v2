#include <MTensor/nn.hpp>
#include <numeric>

namespace mt {
namespace nn {

    UnflattenImpl::UnflattenImpl(Dim dim, Shape shape): _dim(dim), _shape(shape){
        numel = std::accumulate(shape.begin(), shape.end(), 1ll, std::multiplies<Dim>());
    }

    Tensor UnflattenImpl::forward(Tensor input){
        if (input.shape().size() <= _dim || input.shape()[_dim] != numel ){
            throw std::invalid_argument("unflatten():  can't perform the operation on the input");
        }
        auto new_shape = input.shape();
        new_shape.erase(new_shape.begin() + _dim);
        new_shape.insert(new_shape.begin()+_dim, _shape.begin(), _shape.end());
        return input.reshape(new_shape);
    }

    MTENSOR_API std::shared_ptr<Module> Unflatten(Dim dim, Shape shape){
        return std::make_shared<UnflattenImpl>(dim, shape);
    }

}//nn
}//mt
