
#include <MTensor/data.hpp>
#include <MTensor/tensor.hpp>
#include <thread>
#include <thread>
#include <mutex>
#include <queue>
#include <future> 

namespace mt{
namespace data{

namespace fs = std::filesystem;

struct ImageProcessingTask {
    fs::path image_path;
    Tensor label;
    std::pair<int, int> resize_dims;
}; 


void worker_thread_func(
    std::queue<ImageProcessingTask>& task_queue,
    std::mutex& queue_mutex,
    std::condition_variable& queue_cv,
    bool& stop_processing,
    ImageFolderDataset* dataset,
    std::mutex& samples_mutex
) {
    auto& samples = dataset->samples;
    while (true) {
        ImageProcessingTask task;
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            queue_cv.wait(lock, [&task_queue, &stop_processing]{ return !task_queue.empty() || stop_processing; });

            if (stop_processing && task_queue.empty()) {
                break; // Exit if no more tasks and processing is stopped
            }

            task = task_queue.front();
            task_queue.pop();
        }

        try {
            auto img = cimg_library::CImg<float>(task.image_path.string().c_str()).normalize(0.0f, 1.0f);
            if (task.resize_dims.first && task.resize_dims.second) {
                img = img.resize(task.resize_dims.first, task.resize_dims.second);
            }
            
            // Convert to tensor and add to shared samples vector
            std::lock_guard<std::mutex> lock(samples_mutex);
            samples.emplace_back(image_to_tensor(img), task.label);

        } catch (std::exception& e) {
            std::cerr << "Error processing image " << task.image_path << ": " << e.what() << std::endl;
        }
    }
}

    ImageFolderDataset::ImageFolderDataset(const std::string& folder, const std::pair<int,int>& resize, bool vectorize_labels){
        is_label_vectorized = vectorize_labels;

        std::queue<ImageProcessingTask> task_queue;
        std::mutex queue_mutex;
        std::condition_variable queue_cv;
        bool stop_processing = false;
        std::mutex samples_mutex;


        int label_index = 0;
        int num_classes = 0;


        if (is_label_vectorized) {
            for (const auto& entry : fs::directory_iterator(folder)) {
                if (entry.is_directory()) num_classes++;
            }
        }


        unsigned int num_threads = std::thread::hardware_concurrency(); // Use available cores
        if (num_threads == 0) num_threads = 4; // Fallback
        std::vector<std::thread> workers;
        for (unsigned int i = 0; i < num_threads; ++i) {
            workers.emplace_back([
                &samples_mutex,
                this,
                &stop_processing,
                &queue_cv,
                &queue_mutex,
                &task_queue
            ]() { 
            worker_thread_func(
                task_queue,
                queue_mutex,
                queue_cv,
                stop_processing,
                this,
                samples_mutex); 
            });
        }

        for (const auto& entry : fs::directory_iterator(folder)) {
            if (entry.is_directory()) {
                std::string class_name = entry.path().filename().string();
                idx_to_class[label_index] = class_name;

                for (const auto& img_file : fs::directory_iterator(entry.path())) {
                    
                   

                    if (img_file.is_regular_file()) {
                        Tensor label;
                        if (is_label_vectorized) {
                            label = Tensor::zeros({1, (size_t)num_classes});
                            *(label.data_ptr() + label_index) = 1.0f;
                        } else {
                            label = Tensor::empty({1,1});
                            *label.data_ptr() = static_cast<float>(label_index);
                        }

                        // Add task to queue
                        {
                            std::lock_guard<std::mutex> lock(queue_mutex);
                            task_queue.push({img_file.path(), label, resize});
                        }
                        queue_cv.notify_one(); // Notify one waiting worker
                    }
                }
                label_index++;
            }
        }

        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            stop_processing = true;
        }
        queue_cv.notify_all(); 

        for (auto& worker : workers) {
            worker.join();
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