#ifndef GENERAL_H
#define GENERAL_H 
#include <libs/json.hpp>

namespace mt{

class TensorImpl;

namespace utils{

    bool check_json_shape(const nlohmann::json& j, std::vector<int64_t>& shape);
    bool flatten_json_to_buffer(const nlohmann::json& j, float* buffer, int64_t& index);

} //utils

} //mt 


#endif //GENERAL_H