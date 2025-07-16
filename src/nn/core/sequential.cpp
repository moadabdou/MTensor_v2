#include <MTensor/nn.hpp>

namespace mt {
namespace nn {

    SequentialImpl::SequentialImpl(std::initializer_list<std::shared_ptr<Module>> list) {
        for (auto& m : list) {
            module(m);
        };
    }

    Tensor SequentialImpl::forward(Tensor input) {
        for (auto& m : m_direct_children) {
            input = m->forward(input);
        }
        return input;
    }

    MTENSOR_API std::shared_ptr<Module> Sequential(std::initializer_list<std::shared_ptr<Module>> list){
        return std::make_shared<SequentialImpl>(list);
    }

}//nn
}//mt
