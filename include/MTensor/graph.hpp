#ifndef GRAPH_HPP
#define GRAPH_HPP

#include <memory>
#include <libs/json.hpp>
#include <config/mtensor_export.hpp>


using json = nlohmann::json;

//helpers 
void export_as_html(std::ofstream &out, json &data);


namespace mt{

    class TensorImpl;
    class Tensor;
    
namespace graph{

    class MTENSOR_API GradGraph{

    public:

        GradGraph(const std::shared_ptr<TensorImpl>& root);
        GradGraph(const Tensor& root);
        void export_to(const std::string& file) const;

    private:
        TensorImpl* m_root;
    };

}//graph

} //mt

#endif //GRAPH_HPP





