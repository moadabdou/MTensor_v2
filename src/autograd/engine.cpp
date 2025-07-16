#include <unordered_map>
#include <algorithm>
#include <unordered_set>
#include <queue>
#include <stdexcept>
#include <chrono>
#include <MTensor/tensorImpl.hpp>
#include <MTensor/autograd.hpp>

namespace mt {
namespace auto_grad{

    Engine::Engine(const std::shared_ptr<TensorImpl>& root): m_root(root) {} 

    void Engine::sort(){
        topo_sort();
        std::reverse(m_sorted_nodes.begin(), m_sorted_nodes.end());
    }

    void Engine::topo_sort(){
        std::unordered_map<mt::TensorImpl*, int> in_degree;
        std::unordered_map<mt::TensorImpl*, std::vector<mt::TensorImpl*>> reverse_edges;
        std::unordered_set<mt::TensorImpl*> visited;

        std::queue<mt::TensorImpl*> q;

        // Step 1: BFS to build the graph structure
        std::queue<mt::TensorImpl*> build;
        build.push(m_root.get());
        visited.insert(m_root.get());

        while (!build.empty()) {
            mt::TensorImpl* ten = build.front(); build.pop();

            const auto& grad_fn = ten->grad_fn();

            if (!grad_fn) continue; 

            for (auto& input_tensor : grad_fn->operands()) {

                auto parent = input_tensor.get();

                reverse_edges[parent].push_back(ten);
                in_degree[ten]++;

                if (visited.insert(parent).second) {
                    build.push(parent);
                }
            }
        }


        for (mt::TensorImpl* ten : visited) {
            if (in_degree[ten] == 0) {
                q.push(ten);
            }
        }

        while (!q.empty()) {
            mt::TensorImpl* ten = q.front(); q.pop();
            m_sorted_nodes.push_back(ten);

            for (mt::TensorImpl* user : reverse_edges[ten]) {
                in_degree[user]--;
                if (in_degree[user] == 0) {
                    q.push(user);
                }
            }
        }
    }


    void Engine::backward(bool retain_graph){

        if (! m_sorted_nodes[0]->grad_fn()){
            throw std::runtime_error("backward() the grad graph is propably freed, if you want to call backward more than one time use retain_graph = true");
        }

        for (mt::TensorImpl* node : m_sorted_nodes) {
            auto out_grad = node->get_grad();
            if (node->grad_fn()){

                // auto start = std::chrono::high_resolution_clock::now ();

                node->grad_fn()->backward(out_grad);

                // auto end = std::chrono::high_resolution_clock::now ();

                // std::chrono::duration<double,std::milli> d = end - start;
                // std::cout << "back " << node->grad_fn()->name() << " :" << d.count() << "ms\n";

                if ( node != m_sorted_nodes[0] ) {
                    node->set_grad(nullptr);
                }
                if (!retain_graph && node != m_sorted_nodes[0]){
                    node->set_data_ptr(nullptr);
                }
            }
        }
        if (!retain_graph){
            m_sorted_nodes[0]->set_grad_fn(nullptr);
        }
    }
  
}// auto grad
}//mt 

