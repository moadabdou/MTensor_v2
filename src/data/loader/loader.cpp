#include <MTensor/tensor.hpp>
#include <MTensor/data.hpp>
#include <random>
#include <numeric>


namespace mt{
namespace data{


    
    DataLoader::DataLoader(const Dataset& _dataset, int64_t _batch_size, bool _shuffle):
    dataset(_dataset),shuffle(_shuffle), batch_size(_batch_size)
    {
        reset();
    }

    void  DataLoader::reset() {
        indices.resize(dataset.size());
        std::iota(indices.begin(), indices.end(), 0);
        if (shuffle) {
            std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});
        }
        current_idx = 0;
    }

    bool DataLoader::has_next() const {
        return current_idx < dataset.size();
    }


    std::pair<mt::Tensor, mt::Tensor> DataLoader::next() {
        int64_t end = std::min(current_idx + batch_size, dataset.size());
        int64_t actual_batch = end - current_idx;

        std::vector<int64_t> batch_indices;
        batch_indices.assign( indices.begin() + current_idx, indices.begin() + end);

        current_idx = end;
        return dataset.get(batch_indices);
    }

}//data

} //mt