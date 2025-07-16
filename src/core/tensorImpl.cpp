#include <iostream>
#include <random>
#include <stdexcept>
#include <numeric>
#include <dnnl.hpp>
#include <omp.h>
#include <MTensor/tensorImpl.hpp>

namespace mt{

    std::vector<int64_t> row_major_stride( const std::vector<int64_t>& shape){
        std::vector<int64_t> stride(shape.size());
        int64_t strd = 1;

        for(int64_t i = shape.size()-1; i >= 0u ; i--){
            stride[i] = strd;
            strd *= shape[i];
        }
        
        return stride;
    }

    TensorImpl::TensorImpl(const std::vector<int64_t>& shape, bool requires_grad):
    m_requires_grad(requires_grad), m_shape(shape)
    {
        init_tensor_desc();
    }


    TensorImpl::TensorImpl(float* data, const std::vector<int64_t>& shape, bool requires_grad):
    m_requires_grad(requires_grad), m_shape(shape)
    {
        
        init_tensor_desc();
        set_name();

        dnnl::engine eng(dnnl::engine::kind::cpu, 0);
        dnnl::stream strm(eng);

        dnnl::memory::desc md(m_shape, dnnl::memory::data_type::f32, m_stride);

        dnnl::memory src_m(md, eng, data);
        dnnl::memory dst_m(md, eng, m_data.get());

        dnnl::reorder reorder_primtive(src_m, dst_m);

        reorder_primtive.execute(strm, src_m, dst_m);

        strm.wait();

    }

    TensorImpl::TensorImpl(std::function<float(int64_t)> init_fn ,const std::vector<int64_t>& shape, bool requires_grad):
    m_requires_grad(requires_grad), m_shape(shape)
    {
        set_name();
        init_tensor_desc();

        #pragma omp parallel for
        for (int64_t i = 0; i < m_numel ;  i++){
            m_data.get()[i] = init_fn(i);
        }

    }

    TensorImpl::TensorImpl(
        const std::shared_ptr<float>& data, 
        const int64_t& data_offset,
        const std::vector<int64_t>& shape,
        const std::shared_ptr<ops::Operation>& grad_fn, 
        bool requires_grad,
        bool is_contiguous,
        const std::vector<int64_t>& stride
    ):  m_data(data),
        m_data_offset(data_offset),
        m_shape(shape),
        m_grad_fn(grad_fn), 
        m_requires_grad(requires_grad), 
        m_is_contiguous(is_contiguous)
    {
        set_name();
        if (stride.empty()){ //stride is not given so we calculate it along with numel
            init_tensor_desc(false);
        }else {
            m_stride = stride;
            m_numel = std::accumulate(m_shape.begin(), m_shape.end(),1ull, std::multiplies<int64_t>());
        }
    }

    

    std::shared_ptr<TensorImpl> TensorImpl::zeros(const std::vector<int64_t>& shape, bool requires_grad){
        return std::make_shared<TensorImpl>([](int64_t index)->float{ return 0.0f ;}, shape, requires_grad);
    }

    std::shared_ptr<TensorImpl> TensorImpl::ones(const std::vector<int64_t>& shape, bool requires_grad){
        return std::make_shared<TensorImpl>([](int64_t index)->float{ return 1.0f ;}, shape, requires_grad);
    }

    std::shared_ptr<TensorImpl> TensorImpl::randn(const std::vector<int64_t>& shape ,float mean , float stddev, int64_t seed,  bool requires_grad){

        if (seed != -1){
            std::mt19937 gen(seed);

            const auto& stride = row_major_stride(shape);
            const auto& numel  = stride[0] * shape[0];

            std::shared_ptr<float> data_storage( new float[numel], std::default_delete<float[]>());

            #pragma omp parallel
            {
                int thread_id = omp_get_thread_num();
                std::mt19937 gen(seed + thread_id);
                std::normal_distribution<float> normal(mean, stddev);

                #pragma omp for
                for (int64_t i = 0; i < numel ; ++i)
                    data_storage.get()[i] = normal(gen);
            }

            return std::make_shared<TensorImpl>(data_storage, 0, shape, nullptr, requires_grad, true, stride);

        }else {


            std::random_device rd;
            std::mt19937 gen(rd());
            
            std::normal_distribution<float> normal(mean, stddev);

            return std::make_shared<TensorImpl>([&normal, &gen](int64_t index)->float{ return normal(gen) ;}, shape, requires_grad);

        }
        
    }
    std::shared_ptr<TensorImpl> TensorImpl::rand(const std::vector<int64_t>& shape,float lower_bound, float upper_bound, int64_t seed, bool requires_grad){
        if (seed != -1){
            std::mt19937 gen(seed);

            const auto& stride = row_major_stride(shape);
            const auto& numel  = stride[0] * shape[0];

            std::shared_ptr<float> data_storage( new float[numel], std::default_delete<float[]>());

            #pragma omp parallel
            {
                int thread_id = omp_get_thread_num();
                std::mt19937 gen(seed + thread_id);
                std::uniform_real_distribution<float> uniform(lower_bound, upper_bound);

                #pragma omp for
                for (int64_t i = 0; i < numel ; ++i)
                    data_storage.get()[i] = uniform(gen);
            }

            return std::make_shared<TensorImpl>(data_storage, 0, shape, nullptr, requires_grad, true, stride);

        }else {


            std::random_device rd;
            std::mt19937 gen(rd());
            
            std::uniform_real_distribution<float> uniform(lower_bound, upper_bound);;

            return std::make_shared<TensorImpl>([&uniform, &gen](int64_t index)->float{ return uniform(gen) ;}, shape, requires_grad);

        }
    }

    std::shared_ptr<TensorImpl> TensorImpl::detach() const {
        return std::make_shared<TensorImpl>(
            this->m_data,
            this->m_data_offset,
            this->m_shape,
            nullptr,
            false,
            this->m_is_contiguous,
            this->m_stride
        );
    }


    void TensorImpl::init_tensor_desc(bool allocate){

        if (! is_valid_shape(m_shape)){
            throw std::runtime_error("error : invalid shape is given");
        }

        m_stride.resize(m_shape.size());
        int64_t stride = 1;
        m_numel = 1; 

        for(int64_t i = m_shape.size()-1; i >= 0u ; i--){
            m_numel *= m_shape[i];
            m_stride[i] = stride;
            stride *= m_shape[i];
        }

        if(allocate){
            m_data = std::shared_ptr<float>(new float[m_numel], std::default_delete<float[]>());
        }
    }

    int64_t TensorImpl::numel() const{
        return m_numel;
    }

    int64_t TensorImpl::data_offset() const{
        return m_data_offset;
    }

    std::vector<int64_t> TensorImpl::shape() const{
        return m_shape;
    }

    std::vector<int64_t> TensorImpl::stride() const{
        return m_stride;
    }

    bool TensorImpl::requires_grad() const{
        return m_requires_grad;
    }

    void TensorImpl::set_requires_grad(bool value){
        m_requires_grad = value;
    }

    std::shared_ptr<ops::Operation> TensorImpl::grad_fn() const{
        return m_grad_fn;
    }


    bool TensorImpl::is_contiguous() const {
        return m_is_contiguous;
    }

    bool TensorImpl::is_leaf() const {
        if ( ! m_grad_fn )
            return true;
        else
            return false;
    }

    std::shared_ptr<float> TensorImpl::data_ptr() const {
        return m_data;
    }

    bool TensorImpl::is_valid_shape(const std::vector<int64_t>& shape){
        if (shape.empty()) return false;
        for (auto dim : shape) {
            if (dim <= 0) return false;
        }
        return true;
    }

    std::string TensorImpl::get_name() const {
        return m_name;
    }
    void TensorImpl::set_name(std::string name){
        if (name.empty()){
            m_name =  "Tensor#" + std::to_string(reinterpret_cast<uintptr_t>(this));
        }else {
            m_name = name;
        }
    }
    
    const std::shared_ptr<TensorImpl>& TensorImpl::get_grad() const{
        return m_grad;
    }
    void  TensorImpl::set_grad( const std::shared_ptr<TensorImpl>& grad ){
        m_grad = grad;
    }

    void TensorImpl::set_grad_fn( const std::shared_ptr<ops::Operation>& grad_fn){
        m_grad_fn = grad_fn;
    }

    void TensorImpl::set_data_ptr(std::shared_ptr<float> data){
        m_data = data;
    }

}//mt




