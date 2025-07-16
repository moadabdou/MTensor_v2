
#include <MTensor/data.hpp>
#include <MTensor/tensor.hpp>

namespace mt{
namespace data{

    ImageFolderDataset::ImageFolderDataset(const std::string& folder,const std::pair<int,int>& resize, bool vectorize_labels){
        is_label_vectorized =  vectorize_labels;
        namespace fs = std::filesystem;

        int label_index = 0;
        int num_classes = 0;

        if (is_label_vectorized){
            for (const auto& entry : fs::directory_iterator(folder)) {
                if (entry.is_directory()) num_classes++;
            }
        }
        

        for (const auto& entry : fs::directory_iterator(folder)) {
            if (entry.is_directory()) {
                std::string class_name = entry.path().filename().string();
                idx_to_class[label_index] = class_name;
                for (const auto& img_file : fs::directory_iterator(entry.path())) {
                    if (img_file.is_regular_file()) {
                        Tensor label;
                        if (is_label_vectorized){
                            label = Tensor::zeros({1,num_classes});
                            *( label.data_ptr() + label_index) = 1.0f;
                        }else {
                            label = Tensor::empty({1,1});
                            *label.data_ptr() = label_index;
                        }

                        try{

                            auto img = cimg_library::CImg<float>(img_file.path().string().c_str()).normalize(0.0f, 1.0f);
                            if (resize.first && resize.second){
                                img = img.resize(resize.first, resize.second);
                            }
                            samples.emplace_back( image_to_tensor(img), label);

                        }catch(std::exception& e){
                            throw std::runtime_error(std::string(" ImageFolderDataset(): an error occured while trying to import the folder ") + e.what());
                        }
                    }
                }
                label_index++;
            }
        }
    }

    int64_t ImageFolderDataset::size() const {
        return samples.size();
    }

    std::pair<Tensor, Tensor> ImageFolderDataset::get(const std::vector<int64_t>& indices) const {

        std::vector<Tensor> images(indices.size());  
        std::vector<Tensor> labels(indices.size());

        #pragma omp parallel for
        for (int64_t i = 0; i < indices.size(); i++){
            auto sample = samples[indices[i]];
            images[i] = sample.first;
            labels[i] = sample.second;
        }
        
        return {
            Tensor::cat(images, 0),
            Tensor::cat(labels, 0)
        };

    }

    std::string  ImageFolderDataset::label_to_class(const Tensor& label){

        if (label.shape().size() != 2 || label.shape()[0] != 1){
            throw std::invalid_argument(" idx_to_class(): label must be a 2D vector with shape[0] = 1");
        }

        int64_t idx; 
        if (is_label_vectorized){
            int64_t num_classes = label.shape()[1];
            auto data_ptr = label.data_ptr() + label.data_offset();
            for (int64_t i = 0; i < num_classes ; i++){
                if (*(data_ptr + i)) {
                    idx = i; 
                    break;
                }
            }
        }else {
            idx = static_cast<int64_t>(*(label.data_ptr() + label.data_offset()));
        }

        return idx_to_class.at(idx);
    }


    std::map<int64_t, std::string>  ImageFolderDataset::get_labels() const{
        return idx_to_class;
    }




}//data

} //mt