
#include <MTensor/tensor.hpp>
#include <MTensor/autograd.hpp>
#include <MTensor/graph.hpp>
#include <MTensor/utils/general.hpp>

namespace mt {

    Tensor::Tensor(const std::shared_ptr<TensorImpl>& tensor_impl):
    m_tensor_impl(tensor_impl)
    {}

    Tensor::Tensor(float scalar)
    {
        std::shared_ptr<float> data(new float(scalar));
        m_tensor_impl = std::make_shared<TensorImpl>(data, 0, Shape{1}, nullptr, false,true, Shape{1});
    }

    Tensor::Tensor(const json& input){

        std::vector<int64_t> shape; 

        if (!utils::check_json_shape(input, shape))
            throw std::invalid_argument(" can't initilize the tensor from the given input, must be a valid shape");

        int64_t total = 1;
        for (auto dim : shape) total *= dim;

        int64_t index = 0;
        std::shared_ptr<float> buffer( new float[total], std::default_delete<float[]>());

        if (!utils::flatten_json_to_buffer(input, buffer.get(), index)){
            throw std::invalid_argument(" can't initilize the tensor from the given input, must contain valid data");
        }

        m_tensor_impl = std::make_shared<TensorImpl>( buffer, 0, shape, nullptr, false, true);
    }


    //******************************************/
    //                Tensor
    //******************************************/

    //_________________  info  ____________________

    Shape Tensor::shape() const {
        return m_tensor_impl->shape();
    }
    Shape Tensor::stride() const{
        return m_tensor_impl->stride();
    }
    int64_t Tensor::data_offset() const{
        return m_tensor_impl->data_offset();
    }
    int64_t Tensor::numel() const {
        return m_tensor_impl->numel();
    }
    float* Tensor::data_ptr() const{
        return static_cast<float*>(m_tensor_impl->data_ptr().get());
    }
    bool Tensor::is_contiguous() const{
        return m_tensor_impl->is_contiguous();
    }
    bool Tensor::is_leaf() const{
        return m_tensor_impl->is_leaf();
    }
    bool Tensor::is_scalar() const{
        return this->shape() == Shape(this->shape().size(), 1ull);
    }
    std::string Tensor::get_name() const{
        return m_tensor_impl->get_name();
    }
    float Tensor::item() const{
        return *(data_ptr() + data_offset());
    }
    void Tensor::set_name(std::string name) {
        return m_tensor_impl->set_name(name);
    }
    std::shared_ptr<TensorImpl> Tensor::tensor_impl() const{
        return m_tensor_impl;
    }

    
    //_________________  initializers  ____________________


    Tensor Tensor::randn(const Shape& shape, float mean, float stddev, int64_t seed, bool requires_grad) {
        return TensorImpl::randn(shape, mean, stddev, seed, requires_grad);
    }

    Tensor Tensor::rand(const Shape& shape, float lbound, float ubound, int64_t seed,  bool requires_grad) {
        return TensorImpl::rand(shape, lbound, ubound, seed, requires_grad);
    }
    
    Tensor Tensor::fill(const Shape& shape, float value, bool requires_grad) {
        return std::make_shared<TensorImpl>([value](int64_t idx) -> float {
            return value;
        }, shape, requires_grad);
    }

    Tensor Tensor::ones(const Shape& shape, bool requires_grad) {
        return TensorImpl::ones(shape, requires_grad);
    }

    Tensor Tensor::ones_like(const Tensor& tensor, bool requires_grad) {
        return TensorImpl::ones(tensor.shape(), requires_grad);
    }

    Tensor Tensor::zeros(const Shape& shape, bool requires_grad) {
        return TensorImpl::zeros(shape, requires_grad); 
    }
    
    Tensor Tensor::zeros_like(const Tensor& tensor, bool requires_grad) {
        return TensorImpl::zeros(tensor.shape(), requires_grad);
    }

    Tensor Tensor::arange(float start , float end, float step, bool requires_grad) {

        if ( end <= start ){
            throw std::invalid_argument("error: arange() end must > start");
        }

        int64_t numel = static_cast<int64_t>( (end - start) / step ) + 1;
        start -= step;

        std::shared_ptr<float> data_ptr(new float[numel], std::default_delete<float[]>());

        for (int64_t i = 0 ; i < numel;  i++){
            data_ptr.get()[i] = start += step;
        }

        return std::make_shared<TensorImpl>( data_ptr, 0, Shape{numel}, nullptr, requires_grad, true, Shape{1});
    }

    Tensor Tensor::empty(const Shape& shape){
        return std::make_shared<TensorImpl>(shape, false);
    }

    //******************************************/
    //                operations
    //******************************************/

    //________________ view _______________

    Tensor Tensor::expand(const Shape& shape) const {
        ops::Expand expand(shape);
        return expand.forward({m_tensor_impl});
    }
    Tensor Tensor::narrow(Dim dim, Dim start, Dim end, Dim length) const {
        ops::Narrow narrow(dim, start, length);
        return narrow.forward({m_tensor_impl});
    }

    Tensor Tensor::permute(const Shape& permute) const {
        ops::Permute _permute(permute);
        return _permute.forward({m_tensor_impl});
    }

    Tensor Tensor::slice(const sliceList& slice_list) const {
        ops::Slice slice(slice_list);
        return slice.forward({m_tensor_impl});
    }

    Tensor Tensor::operator[](const sliceList& slice_list) const {
        return slice(slice_list);
    }

    Tensor Tensor::squeeze(Dim dim) const {
        ops::Squeeze squeeze(dim);
        return squeeze.forward({m_tensor_impl});
    }

    Tensor Tensor::unsqueeze(Dim dim) const {
        ops::Unsqueeze unsqueeze(dim);
        return unsqueeze.forward({m_tensor_impl});
    }

    Tensor Tensor::transpose(Dim dim_0, Dim dim_1) const {
        ops::Transpose transpose(dim_0, dim_1);
        return transpose.forward({m_tensor_impl});
    }

    Tensor Tensor::view(const Shape& shape) const {
        ops::View view(shape);
        return view.forward({m_tensor_impl});
    }

    Tensor Tensor::reshape(const Shape& shape) const {
        if (this->is_contiguous()) {
            return this->view(shape);
        }else {
            return this->contiguous().view(shape);
        }
    }

    Tensor Tensor::flat() const {
        return this->view({this->numel()});
    }

    
    //________________ arithmetic _______________
    Tensor Tensor::operator+(const Tensor& other) const {

        ops::Add add;
        return add.forward({m_tensor_impl, other.m_tensor_impl});
    }

    Tensor Tensor::operator-(const Tensor& other) const {
        ops::Sub sub;
        return sub.forward({m_tensor_impl, other.m_tensor_impl});
    }

    Tensor Tensor::operator/(const Tensor& other) const {
        ops::Div div;
        return div.forward({m_tensor_impl, other.m_tensor_impl});
    }

    Tensor Tensor::operator*(const Tensor& other) const {
        ops::Mul mul;
        return mul.forward({m_tensor_impl, other.m_tensor_impl});
    }

    Tensor Tensor::operator+(const float& scalar) const {
        return this->linear(1.f, scalar);
    }

    Tensor Tensor::operator-(const float& scalar) const {
        return this->linear(1.f, -scalar); 
    }

    Tensor Tensor::operator/(const float& scalar) const {
        return this->linear(1.f/scalar);
    }

    Tensor Tensor::operator*(const float& scalar) const {
        return this->linear(scalar);
    }


    //NOTE: this is not the final implementation, i will later implement a real in-place operations

    Tensor Tensor::operator+=(const Tensor& other) {

        if (this->requires_grad()) {
            throw std::runtime_error(" error : in-place operations are not allowed on tensors that requires grad. ");
        }

        ops::Add add;
        m_tensor_impl = add.forward({m_tensor_impl, other.m_tensor_impl});
        return *this;
    }

    Tensor Tensor::operator-=(const Tensor& other) {
        if (this->requires_grad()) {
            throw std::runtime_error(" error : in-place operations are not allowed on tensors that requires grad. ");
        }
        ops::Sub sub;
        m_tensor_impl = sub.forward({m_tensor_impl, other.m_tensor_impl});
        return *this;
    }

    Tensor Tensor::operator/=(const Tensor& other) {
        if (this->requires_grad()) {
            throw std::runtime_error(" error : in-place operations are not allowed on tensors that requires grad. ");
        }
        ops::Div div;
        m_tensor_impl = div.forward({m_tensor_impl, other.m_tensor_impl});
        return *this;
    }

    Tensor Tensor::operator*=(const Tensor& other){
        if (this->requires_grad()) {
            throw std::runtime_error(" error : in-place operations are not allowed on tensors that requires grad. ");
        }
        ops::Mul mul;
        m_tensor_impl = mul.forward({m_tensor_impl, other.m_tensor_impl});
        return *this;
    }

    Tensor Tensor::max(const Tensor& other) const {
        ops::Max max;
        return max.forward({m_tensor_impl, other.m_tensor_impl});
    }

    Tensor Tensor::min(const Tensor& other) const {
        ops::Min min;
        return min.forward({m_tensor_impl, other.m_tensor_impl});
    }

    //________________ logic _______________
    Tensor Tensor::operator<(const Tensor& other) const {
        ops::Lt lt;
        return lt.forward({m_tensor_impl, other.m_tensor_impl});
    }

    Tensor Tensor::operator<=(const Tensor& other) const {
        ops::Le le;
        return le.forward({m_tensor_impl, other.m_tensor_impl});
    }

    Tensor Tensor::operator>(const Tensor& other) const {
        ops::Gt gt;
        return gt.forward({m_tensor_impl, other.m_tensor_impl});
    }

    Tensor Tensor::operator>=(const Tensor& other) const {
        ops::Ge ge;
        return ge.forward({m_tensor_impl, other.m_tensor_impl});
    }

    Tensor Tensor::operator==(const Tensor& other) const {
        ops::Eq eq;
        return eq.forward({m_tensor_impl, other.m_tensor_impl});
    }

    Tensor Tensor::operator!=(const Tensor& other) const {
        ops::Ne ne;
        return ne.forward({m_tensor_impl, other.m_tensor_impl});
    }

    //________________ eltwise ____________
    Tensor Tensor::abs() const {
        ops::Abs abs;
        return abs.forward({m_tensor_impl});
    }

    Tensor Tensor::clip(float lbound, float ubound) const {
        ops::Clip clip(ubound, lbound);
        return clip.forward({m_tensor_impl});
    }

    Tensor Tensor::exp() const {
        ops::Exp exp;
        return exp.forward({m_tensor_impl});
    }

    Tensor Tensor::linear(float alpha, float beta) const {
        ops::Linear linear(alpha, beta);
        return linear.forward({m_tensor_impl}); 
    }

    Tensor Tensor::log() const {
        ops::Log log;
        return log.forward({m_tensor_impl}); 
    }

    Tensor Tensor::pow(float exposant, float scale) const {
        ops::Pow pow(exposant, scale);
        return pow.forward({m_tensor_impl}); 
    }

    Tensor Tensor::relu(float alpha) const {
        ops::Relu relu(alpha);
        return relu.forward({m_tensor_impl}); 
    }

    Tensor Tensor::simgoid() const {
        ops::Sigmoid simgoid;
        return simgoid.forward({m_tensor_impl});
    }

    Tensor Tensor::sqrt() const {
        ops::Sqrt sqrt;
        return sqrt.forward({m_tensor_impl});
    }

    Tensor Tensor::tanh() const {
        ops::Tanh tanh;
        return tanh.forward({m_tensor_impl});
    }

    
    //________________ joining ____________
    
    Tensor Tensor::cat(const std::vector<Tensor>& tensors, Dim dim) {
        ops::Cat cat(dim);
        std::vector<std::shared_ptr<TensorImpl>> _tensors(tensors.size());
        for (int64_t i = 0; i < tensors.size() ;  i ++) _tensors[i] = tensors[i].m_tensor_impl;
        return cat.forward(_tensors);
    }
    
    Tensor Tensor::stack(const std::vector<Tensor>& tensors, Dim dim) {
        ops::Stack stack(dim);
        std::vector<std::shared_ptr<TensorImpl>> _tensors(tensors.size());
        for (int64_t i = 0; i < tensors.size() ;  i ++) _tensors[i] = tensors[i].m_tensor_impl;
        return stack.forward(_tensors);
    }

    //________________ linear _____________

    Tensor Tensor::matmul(const Tensor& other) const {
        ops::Matmul matmul;
        return matmul.forward({m_tensor_impl, other.m_tensor_impl});
    }

    //________________ memory _____________ 
    Tensor Tensor::clone() const {
        ops::Clone clone;
        return clone.forward({m_tensor_impl});
    }

    Tensor Tensor::contiguous() const {
        ops::Contiguous contiguous;
        return contiguous.forward({m_tensor_impl});
    }

    //________________ reduction ___________ 
    
    values_indices Tensor::max(Dim dim) const {
        ops::Max_reduction max_r(dim);
        auto res = max_r.forward({m_tensor_impl});
        return { res ,  max_r.indices()};
    }
    
    values_indices Tensor::min(Dim dim) const {
        ops::Min_reduction min_r(dim);
        auto res =  min_r.forward({m_tensor_impl});
        return { res ,  min_r.indices()};
    }

    Tensor Tensor::mean(Dim dim, float eps) const {
        if (dim == EOD){
            return ops::Mean(0).forward({this->flat().m_tensor_impl});
        }
        ops::Mean mean(dim);
        return mean.forward({m_tensor_impl});
    }

    Tensor Tensor::mul(Shape dims) const {
        ops::Mul_reduction mul_r(dims);
        return mul_r.forward({m_tensor_impl});
    }

    Tensor Tensor::norm_lp_power_p_sum(Shape dims, float p, float eps) const {
        ops::Norm_lp_power_p_sum norm_lp_power(dims);
        return norm_lp_power.forward({m_tensor_impl});
    }

    Tensor Tensor::norm_lp_sum(Shape dims, float p, float eps) const {
        ops::Norm_lp_sum norm_lp(dims);
        return norm_lp.forward({m_tensor_impl});
    }

    Tensor Tensor::sum(Shape dims) const {
        if (!dims.size()) {
            dims.resize(this->shape().size());
            std::iota(dims.begin(), dims.end(), 0ull);
        }
        ops::Sum sum(dims);
        return sum.forward({m_tensor_impl});
    }

    //_______________ softmax ____________

    Tensor Tensor::softmax(Dim dim) const {
        ops::Softmax softmax(dim);
        return softmax.forward({m_tensor_impl});
    }

    Tensor Tensor::logsoftmax(Dim dim) const {
        ops::SoftmaxLog softmaxlog(dim);
        return softmaxlog.forward({m_tensor_impl});
    }

    //******************************************/
    //                autograd
    //******************************************/

    Tensor Tensor::grad() const {
        return m_tensor_impl->get_grad();
    }

    // todo: make a wrapper for Operation
    ops::Operation* Tensor::grad_fn() const {
        return m_tensor_impl->grad_fn().get();
    }

    Tensor Tensor::detach() const {
        return m_tensor_impl->detach();
    }

    void Tensor::backward(bool retain_graph) const {

        //not a scalar
        if ( ! this->is_scalar() ) {
            throw std::runtime_error("error: backward() without tensor argements is allowed only for scalars");
        }

        this->backward(Tensor::ones_like(*this), retain_graph);

    } 
    
    void Tensor::backward(const Tensor& grad_out, bool retain_graph) const {

        if ( grad_out.shape() != this->shape() ){
            throw std::runtime_error("error: backward() grad_out must have same shape as the tensor");
        }
        
        m_tensor_impl->set_grad(grad_out.m_tensor_impl);

        auto_grad::Engine grad_eng(m_tensor_impl);
        
        grad_eng.sort();

        grad_eng.backward(retain_graph);
    }

    bool Tensor::requires_grad() const {
        return m_tensor_impl->requires_grad();
    }

    void Tensor::make_requires_grad(){
        m_tensor_impl->set_requires_grad(true);
    }

    graph::GradGraph Tensor::build_graph() const {
        return graph::GradGraph(m_tensor_impl);
    } 



}//mt 



