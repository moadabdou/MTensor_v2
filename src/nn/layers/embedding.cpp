#include <MTensor/nn.hpp>

namespace mt {
namespace nn {

    EmbeddingImpl::EmbeddingImpl(int64_t num_embedding, int64_t embedding_dim){

        embeddings = paramter(Tensor::randn({num_embedding, embedding_dim}, 0.0f, 1.0f, -1, true));

    }


    Tensor EmbeddingImpl::forward(Tensor input){
        if (input.shape().size() != 1){
            throw std::invalid_argument("error: Embedding() input tensor must be 1D tensor of embedding indices!");
        }

        int64_t num_inputs = input.shape()[0];

        std::vector<int64_t> indices(num_inputs);

        for (int64_t i = 0 ; i < num_inputs ; i++){
            indices[i]= static_cast<int64_t>( *(input.data_ptr() + input.data_offset() + i));
        }

        return ops::Embedding(indices).forward({embeddings.tensor_impl()});

    }

    Tensor EmbeddingImpl::weights() const{
        return embeddings;
    }

    MTENSOR_API std::shared_ptr<Module> Embedding(int64_t num_embedding, int64_t embedding_dim){
        return std::make_shared<EmbeddingImpl>(num_embedding, embedding_dim);
    }

}//nn
}//mt
