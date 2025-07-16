#include <fstream>
#include <vector>
#include <stdexcept>
#include <omp.h>
#include <MTensor/tensor.hpp> 

namespace mt {


namespace data{

MTENSOR_API Tensor read_mnist_images(const std::string& filename) {

    std::ifstream file(filename, std::ios::binary);

    if (!file) throw std::runtime_error("Cannot open " + filename);

    int32_t magic_number = 0, num_images = 0, rows = 0, cols = 0;

    file.read(reinterpret_cast<char*>(&magic_number), 4);
    file.read(reinterpret_cast<char*>(&num_images), 4);
    file.read(reinterpret_cast<char*>(&rows), 4);
    file.read(reinterpret_cast<char*>(&cols), 4);

    // Big endian fix
    #if defined(_MSC_VER)
        magic_number = _byteswap_ulong(magic_number);
        num_images = _byteswap_ulong(num_images);
        rows = _byteswap_ulong(rows);
        cols = _byteswap_ulong(cols);
    #else
        magic_number = __builtin_bswap32(magic_number);
        num_images = __builtin_bswap32(num_images);
        rows = __builtin_bswap32(rows);
        cols = __builtin_bswap32(cols);
    #endif

    int64_t total = int64_t(num_images) * rows * cols;
    std::vector<uint8_t> buffer(total);
    file.read(reinterpret_cast<char*>(buffer.data()), total);

    auto images = Tensor::empty({num_images, 1, rows, cols}); // shape: N,C,H,W

    float* data_ptr = images.data_ptr();

    #pragma omp parallel for
    for (int64_t i = 0; i < total; ++i) {
        data_ptr[i] = static_cast<float>(buffer[i]) / 255.0f;  // normalize to [0, 1]
    }

    return images;
}

MTENSOR_API Tensor read_mnist_labels(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open " + filename);

    int32_t magic_number = 0, num_labels = 0;

    file.read(reinterpret_cast<char*>(&magic_number), 4);
    file.read(reinterpret_cast<char*>(&num_labels), 4);


    #if defined(_MSC_VER)
        magic_number = _byteswap_ulong(magic_number);
        num_labels = _byteswap_ulong(num_labels);
    #else
        magic_number = __builtin_bswap32(magic_number);
        num_labels = __builtin_bswap32(num_labels);
    #endif


    std::vector<uint8_t> buffer(num_labels);
    file.read(reinterpret_cast<char*>(buffer.data()), num_labels);

    auto labels = Tensor::empty({num_labels});

    float* data_ptr = reinterpret_cast<float*>(labels.data_ptr());

    #pragma omp parallel for
    for (int i = 0; i < num_labels; ++i) {
        data_ptr[i] = static_cast<float>(buffer[i]);
    }
    return labels;
}


MTENSOR_API Tensor read_mnist_labels_as_vectors(const std::string& filename, int num_classes){
    std::ifstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open " + filename);

    int32_t magic_number = 0, num_labels = 0;

    file.read(reinterpret_cast<char*>(&magic_number), 4);
    file.read(reinterpret_cast<char*>(&num_labels), 4);


    #if defined(_MSC_VER)
        magic_number = _byteswap_ulong(magic_number);
        num_labels = _byteswap_ulong(num_labels);
    #else
        magic_number = __builtin_bswap32(magic_number);
        num_labels = __builtin_bswap32(num_labels);
    #endif


    std::vector<uint8_t> buffer(num_labels);
    file.read(reinterpret_cast<char*>(buffer.data()), num_labels);

    auto labels = Tensor::zeros({num_labels, num_classes});

    float* data_ptr = reinterpret_cast<float*>(labels.data_ptr());

    #pragma omp parallel for
    for (int i = 0; i < num_labels; ++i) {
        data_ptr[i * num_classes + buffer[i]] = 1.0f;
    }
    return labels;
}

}//data

} // namespace mt
