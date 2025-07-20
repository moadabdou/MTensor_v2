#ifndef DATA_HPP
#define DATA_HPP


#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <utility>
#include <map>
#include <config/mtensor_export.hpp>
#include <MTensor/tensor.hpp>

#ifdef mt_use_img
#include <libs/CImg.h>
using namespace cimg_library;
#endif

namespace mt{

namespace data{

    /************************************************/
    //                  image tools                 //
    /************************************************/

    #ifdef mt_use_img
    /**
     * this function will return a tensor of shape (1,C,W,H) 
     */
    MTENSOR_API Tensor image_to_tensor(const cimg_library::CImg<float>& img);

    /**
     * this function will only accept a tensor of shape (1,C,W,H)
     */
    MTENSOR_API cimg_library::CImg<float> tensor_to_image(const Tensor& tensor);

    #endif

    /************************************************/
    //                  MNIST tools                 //
    /************************************************/

    MTENSOR_API Tensor read_mnist_images(const std::string& filename);
    MTENSOR_API Tensor read_mnist_labels(const std::string& filename);
    MTENSOR_API Tensor read_mnist_labels_as_vectors(const std::string& filename, int num_classes);


    /************************************************/
    //                    DATASETS                  //
    /************************************************/

    MTENSOR_API int64_t compare_output_target(const Tensor& output, const Tensor& target);

    class MTENSOR_API Dataset {
    public:
        virtual int64_t size() const = 0;
        virtual std::pair<Tensor, Tensor> get(const std::vector<int64_t>& indices) const = 0;
        virtual ~Dataset() = default;
    };

    class MTENSOR_API MNISTDataset : public Dataset {
    private:
        Tensor images; 
        Tensor labels;
    public:
        MNISTDataset(const std::string& image_path, const std::string& label_path, uint32_t vectorize_labels = 0);
        int64_t size() const override;
        std::pair<Tensor, Tensor> get(const std::vector<int64_t>& indices) const override ;
        std::string vectoried_label_to_number(const Tensor& label);
    };

    #ifdef mt_use_img

    class MTENSOR_API ImageFolderDataset : public Dataset {
    public:
        std::vector<std::pair<Tensor,Tensor>> samples;  // image, label
        std::map<int64_t, std::string> idx_to_class;    //label_idx , class_name
        bool is_label_vectorized;
        ImageFolderDataset(const std::string& folder, const std::pair<int,int>& resize = {}, bool vectorize_labels = true);
        int64_t size() const override;
        std::pair<Tensor, Tensor> get(const std::vector<int64_t>& indices) const override ;
        std::string label_to_class(const Tensor& label);
        std::map<int64_t, std::string> get_labels() const;
    };

    #endif

    class MTENSOR_API DataLoader {
    private:
        const Dataset& dataset;
        int64_t batch_size;
        bool shuffle;
        std::vector<int64_t> indices;
        int64_t current_idx = 0;

    public:
        DataLoader(const Dataset& ds, int64_t bs, bool shuf = true);

        void reset();

        bool has_next() const;

        std::pair<mt::Tensor, mt::Tensor> next();
    };

}//data

} //mt

#endif //DATA_HPP