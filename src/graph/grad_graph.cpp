#include <unordered_set>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <MTensor/tensorImpl.hpp>
#include <MTensor/graph.hpp>
#include <MTensor/tensor.hpp>

#if defined(_WIN32)
#include <windows.h>
#include <heapapi.h>
#elif defined(__APPLE__)
#include <malloc/malloc.h>
#elif defined(__linux__)
#include <malloc.h>
#endif




////////////////////////////////////
 
// TODO : move allocated mem to utils 
 
//////////////////////////////

struct TensorDetails{
    std::string shape;
    std::string stride;
    bool is_contiguous;
    int64_t numel;
    bool requires_grad;
    std::string grad_fn;
    std::string mem_size;
    uintptr_t memory_ptr;
};

struct GeneralDetails{
    std::string total_allocated_memory;
    std::string memory_saved_by_tensor_view;
    int64_t number_of_tensors = 0;
    int64_t number_of_ops = 0;
    int64_t number_of_tensors_require_grad = 0;
};

TensorDetails get_tensor_details(mt::TensorImpl* tensor);
std::string shape_to_str(const std::vector<int64_t> &shape);
std::string allocated_mem_str(void *ptr);
size_t allocated_mem(void *ptr);
std::string mem_to_human(size_t size_in_bytes);

namespace mt
{

    namespace graph
    {

        GradGraph::GradGraph(const std::shared_ptr<TensorImpl> &root) : m_root(root.get()) {}
        GradGraph::GradGraph(const Tensor& root) : m_root(root.tensor_impl().get()) {}

        void GradGraph::export_to(const std::string &file) const
        {

            json j;
            std::unordered_set<TensorImpl *> visited;
            std::unordered_set<void *> calculated_mem;
            std::vector<json> nodes, edges;
            GeneralDetails general_details;
            size_t total_memory = 0;
            size_t total_memory_all_nodes = 0;

            std::function<void(TensorImpl *)> dfs = [&](TensorImpl *node)
            {
                if (visited.count(node))
                    return;
                visited.insert(node);

                TensorDetails tensor_details = get_tensor_details(node);
                
                general_details.number_of_tensors ++;

                if (node->requires_grad()) 
                    general_details.number_of_tensors_require_grad ++;

                auto tensor_data_ptr = reinterpret_cast<void*>(node->data_ptr().get());
                total_memory_all_nodes += allocated_mem(tensor_data_ptr);
                if (!calculated_mem.count(tensor_data_ptr)) {
                    total_memory += allocated_mem(tensor_data_ptr);
                    calculated_mem.insert(tensor_data_ptr);
                }

                std::string tensor_id = std::to_string(reinterpret_cast<uintptr_t>(node));

                json nd = {
                    {"data", {
                        {"id", tensor_id}, 
                        {"label", node->get_name() + "\n" + tensor_details.shape}, 
                        {"details" , {
                            {"shape" , tensor_details.shape},
                            {"stride", tensor_details.stride},
                            {"number of elements", tensor_details.numel},
                            {"is contiguous", tensor_details.is_contiguous},
                            {"requires grad", tensor_details.requires_grad},
                            {"grad function", tensor_details.grad_fn},
                            {"allocated memory", tensor_details.mem_size},
                            {"memory pointer", tensor_details.memory_ptr}
                        }}
                    }},
                    {"classes", "tensor"}
                }; 

                if (node->get_grad()){
                    const auto& grad = node->get_grad();
                    TensorDetails grad_details = get_tensor_details(grad.get());
                    
                    auto grad_data_ptr = grad->data_ptr().get();
                    
                    total_memory_all_nodes += allocated_mem(grad_data_ptr);
                    if (!calculated_mem.count(grad_data_ptr)) {
                        total_memory += allocated_mem(grad_data_ptr);
                        calculated_mem.insert(grad_data_ptr);
                    }

                    nd["data"]["grad_details"] = {
                        {"id",  std::to_string(reinterpret_cast<uintptr_t>(grad.get())) },
                        {"label", grad->get_name() + "\n" + grad_details.shape}, 
                        {"details" , {
                            {"shape" , grad_details.shape},
                            {"stride", grad_details.stride},
                            {"number of elements", grad_details.numel},
                            {"is contiguous", grad_details.is_contiguous},
                            {"requires grad", grad_details.requires_grad},
                            {"grad function", grad_details.grad_fn},
                            {"allocated memory", grad_details.mem_size},
                            {"memory pointer", grad_details.memory_ptr}
                        }}
                    };
                }

                nodes.push_back(nd);

                if (!node->grad_fn())
                    return;

                general_details.number_of_ops ++;
                std::string op_id = std::to_string(reinterpret_cast<uintptr_t>(node->grad_fn().get()));
                nodes.push_back({{"data", {{"id", op_id}, {"label", node->grad_fn()->name()}}}, {"classes", "op"}});
                edges.push_back({{"data", {{"source", tensor_id}, {"target", op_id}}}, {"classes", "edge"}});

                for (auto operand : node->grad_fn()->operands())
                {
                    std::string operand_id = std::to_string(reinterpret_cast<uintptr_t>(operand.get()));
                    edges.push_back({{"data", {{"source", op_id}, {"target", operand_id}}}});
                    dfs(operand.get());
                }
            };

            dfs(m_root);

            general_details.total_allocated_memory = mem_to_human(total_memory);
            general_details.memory_saved_by_tensor_view = mem_to_human(total_memory_all_nodes - total_memory);

            j["graph_data"]= {
                {"nodes", nodes},
                {"edges", edges}
            };
            
            j["general"] = {
                {"total allocated memory", general_details.total_allocated_memory},
                {"total memory saved by tensor view", general_details.memory_saved_by_tensor_view},
                {"number of tensors" , general_details.number_of_tensors},
                {"number of tensors require grad" , general_details.number_of_tensors_require_grad},
                {"number of ops" , general_details.number_of_ops}
            };

            std::ofstream out(file);
            export_as_html(out, j);
        }

    } // graph
} // mt




TensorDetails get_tensor_details(mt::TensorImpl* tensor){
    return {
        shape_to_str(tensor->shape()),
        shape_to_str(tensor->stride()),
        tensor->is_contiguous(),
        tensor->numel(),
        tensor->requires_grad(),
        tensor->grad_fn()? tensor->grad_fn()->name() : "Init",
        allocated_mem_str( static_cast<void*>(tensor->data_ptr().get()) ),
        reinterpret_cast<uintptr_t>(tensor->data_ptr().get())
    };
}


std::string shape_to_str(const std::vector<int64_t> &shape)
{
    std::string res = "(";
    for (int i = 0; i < shape.size(); i++)
    {
        res += std::to_string(shape[i]) + (i < (shape.size() - 1) ? ", " : " )");
    }
    return res;
}


size_t allocated_mem(void *ptr) {
    if (ptr == nullptr) {
        return 0;
    }

    size_t size_in_bytes = 0;

#if defined(_WIN32)
    // On Windows, _msize is a common function, but HeapSize is more robust.
    // We need to get the handle to the process heap first.
    HANDLE process_heap = GetProcessHeap();
    if (process_heap != NULL) {
        size_in_bytes = HeapSize(process_heap, 0, ptr);
    }
#elif defined(__APPLE__)
    size_in_bytes = malloc_size(ptr);
#elif defined(__linux__)
    size_in_bytes = malloc_usable_size(ptr);
#else
    // No standard way to get the allocated size on other platforms
    return 0;
#endif

    return size_in_bytes;
}

std::string mem_to_human(size_t size_in_bytes){
    
    if (size_in_bytes == 0) {
        return "0 B";
    }

    const char* suffixes[] = {"B", "KB", "MB", "GB", "TB"};
    int suffix_index = 0;
    double size = static_cast<double>(size_in_bytes);

    while (size >= 1024 && suffix_index < 4) {
        size /= 1024;
        suffix_index++;
    }

    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << size << " " << suffixes[suffix_index];
    return ss.str();
}

std::string allocated_mem_str(void *ptr) {
    return mem_to_human(allocated_mem(ptr));
}
