
#include <fstream>
#include <MTensor/nn.hpp>

namespace mt {
namespace nn {

        MTENSOR_API void save_model(Module& model, save_states& states,const std::string& path){
            std::ofstream out(path, std::ios::binary);
            out.write( reinterpret_cast<char*>(&states) , sizeof(save_states));
            auto paramters = model.paramters();
            auto auxiliaries = model.auxiliaries();

            for (auto& p : paramters){
                out.write(reinterpret_cast<char*>(p.data_ptr() + p.data_offset()), p.numel()*sizeof(float));
            }
            for (auto& a : auxiliaries){
                out.write(reinterpret_cast<char*>(a.data_ptr() + a.data_offset()), a.numel()*sizeof(float));
            }

            out.close();
        }

        MTENSOR_API save_states load_model(Module& model, const std::string& path){

            save_states states;

            std::ifstream in(path, std::ios::binary);

            if(!in){
                throw std::runtime_error("load_model(): can't load model, file does not exist");
            }

            in.read( reinterpret_cast<char*>(&states) , sizeof(save_states));

            auto paramters = model.paramters();
            auto auxiliaries = model.auxiliaries();

            for (auto& p : paramters){
                in.read(reinterpret_cast<char*>(p.data_ptr()), p.numel()*sizeof(float));
            }

            for (auto& a : auxiliaries){
                in.read(reinterpret_cast<char*>(a.data_ptr()), a.numel()*sizeof(float));
            }

            in.close();

            return states;

        }

}//nn
}//mt
