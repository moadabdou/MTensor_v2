
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
    std::string  MNISTDataset::vectoried_label_to_number(const Tensor& v_label){

        if (v_label.shape().size() != 2 || v_label.shape()[0] != 1){
            throw std::invalid_argument(" vectoried_label_to_number(): v_label must be a 2D vector with shape[0] = 1");
        }

        int64_t number; 
        if (v_label.shape()[1] > 1){
            int64_t num_classes = v_label.shape()[1];
            auto data_ptr = v_label.data_ptr() + v_label.data_offset();
            for (int64_t i = 0; i < num_classes ; i++){
                if (*(data_ptr + i)) {
                    number = i; 
                    break;
                }
            }
        }else {
            number = static_cast<int64_t>(*(v_label.data_ptr() + v_label.data_offset()));
        }

        return std::to_string(number);

    }



}//data

} //mt