
#include <MTensor/data.hpp>
#include <MTensor/tensor.hpp>

namespace mt{
namespace data{


    
    MNISTDataset::MNISTDataset(const std::string& image_path, const std::string& label_path, uint32_t vectorize_labels){
        images = read_mnist_images(image_path);
        labels = vectorize_labels ?  read_mnist_labels_as_vectors(label_path, vectorize_labels) : read_mnist_labels(label_path);
    }

    int64_t MNISTDataset::size() const {
        return images.shape()[0];
    }

    std::pair<Tensor, Tensor> MNISTDataset::get(const std::vector<int64_t>& indices) const {

        ops::Embedding emb(indices);

        return {
            emb.forward({images.tensor_impl()}),
            emb.forward({labels.tensor_impl()})
        };

    }



}//data

} //mt