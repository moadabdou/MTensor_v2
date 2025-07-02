#ifndef GRAPH_H
#define GRAPH_H

#include <memory>
#include <libs/json.hpp>
#include <config/mtensor_export.hpp>


using json = nlohmann::json;

//helpers 
void export_as_html(std::ofstream &out, json &data);


namespace mt{

    class TensorImpl;
    
namespace graph{

    class MTENSOR_API GradGraph{

    public:

        GradGraph(const std::shared_ptr<TensorImpl>& root);
        void export_to(const std::string& file) const;

    private:
        TensorImpl* m_root;
    };

}//graph

} //mt

#endif //GRAPH_H





