#include <MTensor/nn.hpp>

namespace mt {
namespace nn {

    std::shared_ptr<Module> Module::module(const std::shared_ptr<Module>& module){
        m_direct_children.push_back(module);
        return module;
    }

    Tensor Module::operator()(Tensor input){
        return forward( input );
    }

    Tensor Module::paramter(const Tensor& prm){
        if (!prm.is_contiguous()){
            throw std::invalid_argument(" Module.paramter(): non contiguous Tensors are not allowed as paramters as optimizers dont support them (they perform eltwise operations ) !");
        }
        m_direct_paramters.push_back(prm);
        return prm;
    }

    Tensor Module::auxiliary(const Tensor& aux){
        if (!aux.is_contiguous()){
            throw std::invalid_argument(" Module.auxiliary(): non contiguous Tensors are not allowed as auxiliarys as optimizers dont support them (they perform eltwise operations ) !");
        }
        m_direct_auxiliaries.push_back(aux);
        return aux;
    }

    void Module::auxiliary(const std::vector<Tensor>& aux){
        m_direct_auxiliaries.insert(m_direct_auxiliaries.end(), aux.begin(), aux.end());
    }

    std::vector<Tensor>  Module::paramters() const{
        std::vector<Tensor> collector;
        _parameters(collector);
        return collector;
    }

    std::vector<Tensor>  Module::auxiliaries() const{
        std::vector<Tensor> collector;
        _auxiliaries(collector);
        return collector;
    }

    void Module::_parameters( std::vector<Tensor>& collector ) const{

        collector.insert(collector.end(), m_direct_paramters.begin(), m_direct_paramters.end());
        
        for (const auto& m : m_direct_children){
            m->_parameters(collector);   
        }
    }

    void Module::_auxiliaries( std::vector<Tensor>& collector ) const{
        collector.insert(collector.end(), m_direct_auxiliaries.begin(), m_direct_auxiliaries.end());
        for (const auto& m : m_direct_children){
            m->_auxiliaries(collector);   
        }
    }

}//nn
}//mt
