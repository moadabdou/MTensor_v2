#include <stdexcept>
#include <numeric>
#include <iostream>
#include <MTensor/tensorImpl.hpp>
#include <MTensor/ops.hpp>


///////////////////////////////////////////
//  TODO : use dnnl's concat operation   //
//////////////////////////////////////////


namespace mt {
namespace ops{

    int64_t Cat::count = 0;

    Cat::Cat(const int64_t& dim , bool inc_counter )
    {
        if (dim < 0 ){
            throw std::invalid_argument(
                "error: Cat() invalide dim was passed"
            );
        }
        m_dim = dim;
        if(inc_counter){
            m_name = "Cat"+std::to_string(count);
            count++;
        }
    } 
 
    std::shared_ptr<TensorImpl> Cat::forward(const std::vector<std::shared_ptr<TensorImpl>>& operands) {

        const auto& op_ref = operands[0];
        const auto& in_shape_size = op_ref->shape().size();
        int64_t m_dim_indice = 0;

        if (m_dim >= in_shape_size) {
            throw std::invalid_argument(
                "error: Cat() m_dim is out of range of the operands"
            );
        }

        for(const auto& el : operands){

            if (in_shape_size != el->shape().size() ) {
                throw std::invalid_argument(
                    "error: Cat() all operands must have the same shape size"
                );
            }

            for(int64_t i = 0; i < in_shape_size; i++){
                if(i != m_dim && el->shape()[i] != op_ref->shape()[i]){
                    throw std::invalid_argument(
                        "error: Cat() all operands must have shape of equal dims except m_dim"
                    );
                }

                if ( i == m_dim){
                    m_dim_indice += el->shape()[i];
                }

            }

        }  

        auto out_shape = op_ref->shape();
        out_shape[m_dim] = m_dim_indice;
        const auto& out_stride = row_major_stride(out_shape);

        const int64_t out_numel = out_shape[0] * out_stride[0];

        std::shared_ptr<float> out_data(new float[out_numel], std::default_delete<float[]>());


        int64_t lower_bound = 0;

        
        dnnl::engine eng(dnnl::engine::kind::cpu, 0);

        for (const auto& op :  operands){

            dnnl::stream strm(eng);

            const auto& op_shape = op->shape();

            dnnl::memory::desc src_md(op_shape , dnnl::memory::data_type::f32, op->stride());
            dnnl::memory::desc dst_md(op_shape , dnnl::memory::data_type::f32, out_stride);

            dnnl::memory src_m(src_md, eng, op->data_ptr().get() + op->data_offset());
            dnnl::memory dst_m(dst_md, eng, out_data.get() + lower_bound * out_stride[m_dim]);

            dnnl::reorder reoder_primitive(src_m, dst_m);

            reoder_primitive.execute(strm, src_m, dst_m);
            strm.wait();     

            lower_bound += op_shape[m_dim];

        }

        std::shared_ptr<Operation> grad_fn = nullptr;
        bool requires_grad = false;

        // Construct an array of operands with requires_grad() == true
        std::vector<std::shared_ptr<TensorImpl>> operands_require_grad;
        for (const auto& op : operands) {
            if (op->requires_grad()) {
                operands_require_grad.push_back(op);
            }
        }

        if (!operands_require_grad.empty()){
            
            requires_grad = true; 
            grad_fn = std::make_shared<Cat>(m_dim, true);
            grad_fn->set_operands(operands_require_grad);

        }

        return std::make_shared<TensorImpl>(out_data, 0, out_shape , grad_fn , requires_grad, true , out_stride);
    }  

    void Cat::backward(const std::shared_ptr<TensorImpl>& diff_loss_out){
//if (diff_loss_out->requires_grad()) std::cout << m_name ; 
        dnnl::engine engine(dnnl::engine::kind::cpu, 0);
        dnnl::stream engine_stream(engine);
        int64_t lower_bound = 0;
        Contiguous contiguous;

        
        auto slice_list = sliceList( diff_loss_out->shape().size(), {0 , EOD});
        
        for (auto& operand :  m_operands){
            
            auto& x = operand;

            if (! x->requires_grad()) continue;

            slice_list[m_dim] = {lower_bound, lower_bound + x->shape()[m_dim]};

            lower_bound += x->shape()[m_dim];

            Slice slice(slice_list);

            auto diff_src =  contiguous.forward({slice.forward({diff_loss_out})});

            
            {
            if (x->get_grad()){

                auto diff_src_md = dnnl::memory::desc(diff_src->shape() , dnnl::memory::data_type::f32, diff_src->stride());
                dnnl::memory diff_src_mem(diff_src_md, engine, diff_src->data_ptr().get() + diff_src->data_offset());
                accumulate(
                    diff_src_mem,
                    x->get_grad(),
                    engine,
                    engine_stream
                );
            }else {
                x->set_grad(diff_src);
            }
            }
        }

    }

}//ops
}//mt