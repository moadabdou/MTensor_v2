#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <MTensor/tensorImpl.hpp>
#include <MTensor/ops.hpp>

namespace mt
{
    namespace ops
    {

        int64_t Mean::count = 0;

        Mean::Mean(int64_t dim, float eps, bool inc_counter) : m_eps(eps), m_dim(dim)

        {

            if (inc_counter)
            {
                m_name = "Mean" + std::to_string(count);
                count++;
            }
        }

        std::shared_ptr<TensorImpl> Mean::forward(const std::vector<std::shared_ptr<TensorImpl>> &operands)
        {

            const auto &in_tensor = operands[0];

            if (in_tensor->shape()[m_dim] == 1)
            {
                return in_tensor;
            }

            const auto &in_shape = in_tensor->shape();
            const auto &src_stride = in_tensor->stride();

            if (m_dim >= in_shape.size())
            {
                throw std::invalid_argument("error: Mean() dim is out of range of in_tensor dims");
            }

            std::vector<int64_t> out_shape = in_shape;

            out_shape[m_dim] = 1;

            const auto &dst_stride = row_major_stride(out_shape);

            dnnl::memory::desc src_md(in_shape, dnnl::memory::data_type::f32, src_stride);
            dnnl::memory::desc dst_md(out_shape, dnnl::memory::data_type::f32, dst_stride);

            dnnl::engine eng(dnnl::engine::kind::cpu, 0);

            dnnl::reduction::primitive_desc reduction_desc(
                eng,
                dnnl::algorithm::reduction_mean,
                src_md,
                dst_md,
                0.0f,
                m_eps);

            const auto &data_storage = custom_reduction_op(
                in_tensor,
                reduction_desc,
                eng,
                src_md,
                dst_md);

            std::shared_ptr<Operation> grad_fn = nullptr;
            bool requires_grad = false;

            if (in_tensor->requires_grad())
            {
                requires_grad = true;
                grad_fn = std::make_shared<Mean>(m_dim, m_eps, true);
                grad_fn->set_operands({in_tensor});
            }

            return std::make_shared<TensorImpl>(data_storage, 0, out_shape, grad_fn, requires_grad, true, dst_stride);
        }

        void Mean::backward(const std::shared_ptr<TensorImpl> &diff_loss_out)
        {
            // if (diff_loss_out->requires_grad()) std::cout << m_name ;
            dnnl::engine engine(dnnl::engine::kind::cpu, 0);
            dnnl::stream engine_stream(engine);

            auto &x = m_operands[0];

            if (!x->requires_grad())
                return;

            Expand expand(x->shape());
            Linear linear(1.0f / x->shape()[m_dim], 0.0f);

            auto diff_src = linear.forward({expand.forward({diff_loss_out})});

            
            {
            if (x->get_grad())
            {

                auto diff_src_md = dnnl::memory::desc(diff_src->shape(), dnnl::memory::data_type::f32, diff_src->stride());
                dnnl::memory diff_src_mem(diff_src_md, engine, diff_src->data_ptr().get() + diff_src->data_offset());
                accumulate(
                    diff_src_mem,
                    x->get_grad(),
                    engine,
                    engine_stream);
            }
            else
            {
                x->set_grad(diff_src);
            }
            }
        }

    } // ops
} // mt