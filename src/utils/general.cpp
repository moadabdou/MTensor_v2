
#include <MTensor/tensor.hpp>
#include <MTensor/utils/general.hpp>
#include <vector>
#include <cstddef>
#include <stdexcept>


namespace mt {
namespace utils {
    

bool check_json_shape(const nlohmann::json& j, std::vector<int64_t>& shape) {
    if (j.is_array()) {
        int64_t len = j.size();
        shape.push_back(len);

        if (len == 0)
            return true;

        std::vector<int64_t> subshape;
        if (!check_json_shape(j[0], subshape)) return false;

        for (int64_t i = 1; i < len; ++i) {
            std::vector<int64_t> tmp;
            if (!check_json_shape(j[i], tmp) || tmp != subshape)
                return false;
        }

        shape.insert(shape.end(), subshape.begin(), subshape.end());
        return true;
    } else if (j.is_number()) {
        return true;
    }

    return false;
}

bool flatten_json_to_buffer(const nlohmann::json& j, float* buffer, int64_t& index) {
    if (j.is_number()) {
        buffer[index++] = j.get<float>();
    } else if (j.is_array()) {
        for (const auto& item : j) {
            if (!flatten_json_to_buffer(item, buffer, index))
                return false;
        }
    } else {
        return false;
    }
    return true;
}

}//utils
}//mt