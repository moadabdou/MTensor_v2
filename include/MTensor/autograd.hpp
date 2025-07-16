#ifndef AUTO_GRAD_HPP
#define AUTO_GRAD_HPP

#include <config/mtensor_export.hpp>
#include <vector>
#include <memory>

namespace mt {
    class TensorImpl;
namespace auto_grad{

    class MTENSOR_API Engine{
    
    public:

        Engine(const std::shared_ptr<TensorImpl>& m_root);
        void sort();
        void backward(bool retain_grad = false);

    private:


        void topo_sort();
        std::shared_ptr<TensorImpl> m_root;
        std::vector<TensorImpl*> m_sorted_nodes;
        std::unordered_map<int, std::vector<mt::TensorImpl*>> m_nodes_by_level;
        int m_max_level = 0;

    };


}// auto grad
}//mt 


#endif  //AUTO_GRAD_HPP