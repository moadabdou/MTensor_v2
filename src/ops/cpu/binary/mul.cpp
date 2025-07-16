#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <MTensor/tensorImpl.hpp>
#include <MTensor/ops.hpp>
#include <MTensor/utils/braodcast.hpp>

namespace mt
{
    namespace ops
    {

        int64_t Mul::count = 0;

        Mul::Mul(bool inc_counter)
        {
            if (inc_counter)
            {
                m_name = "Mul" + std::to_string(count);
                count++;
            }
        }

        std::shared_ptr<TensorImpl> Mul::forward(const std::vector<std::shared_ptr<TensorImpl>> &operands)
        {

            try
            {

                auto [in_tensor_0, in_tensor_1] = utils::broadcast({operands[0], operands[1]});

                const auto &in_shape = in_tensor_0->shape();
                const auto &src_0_stride = in_tensor_0->stride();
                const auto &src_1_stride = in_tensor_1->stride();
                const auto &dst_stride = row_major_stride(in_shape);

                dnnl::memory::desc src_0_md(in_shape, dnnl::memory::data_type::f32, src_0_stride);
                dnnl::memory::desc src_1_md(in_shape, dnnl::memory::data_type::f32, src_1_stride);
                dnnl::memory::desc dst_md(in_shape, dnnl::memory::data_type::f32, dst_stride);

                dnnl::engine eng(dnnl::engine::kind::cpu, 0);

                dnnl::binary::primitive_desc binary_desc(
                    eng,
                    dnnl::algorithm::binary_mul,
                    src_0_md,
                    src_1_md,
                    dst_md);

                const auto &data_storage = custom_binary_op(
                    in_tensor_0,
                    in_tensor_1,
                    binary_desc,
                    eng,
                    src_0_md,
                    src_1_md,
                    dst_md);

                std::shared_ptr<Operation> grad_fn = nullptr;
                bool requires_grad = false;

                if (in_tensor_0->requires_grad() || in_tensor_1->requires_grad())
                {
                    requires_grad = true;
                    grad_fn = std::make_shared<Mul>(true);
                    grad_fn->set_operands({in_tensor_0, in_tensor_1});
                }

                return std::make_shared<TensorImpl>(data_storage, 0, in_shape, grad_fn, requires_grad, true, dst_stride);
            }
            catch (std::exception &e)
            {
                throw std::runtime_error(
                    std::string("error : Mul() ") + e.what());
            }
        }

        void Mul::backward(const std::shared_ptr<TensorImpl> &diff_loss_out)
        {
            dnnl::engine engine(dnnl::engine::kind::cpu, 0);
            dnnl::stream engine_stream(engine);

            Mul mul;

            auto &x = m_operands[0];
            auto &y = m_operands[1];

            
            {
            if (x->requires_grad())
            {
                auto y_prev_grad = y->requires_grad();
                y->set_requires_grad(false);

                auto diff_loss_x = mul.forward({y, // diff_loss_out_x
                                                diff_loss_out});
                y->set_requires_grad(y_prev_grad);

                if (!x->get_grad())
                {

                    x->set_grad(diff_loss_x);
                }
                else
                {

                    dnnl::memory diff_loss_x_mem(
                        {diff_loss_x->shape(), dnnl::memory::data_type::f32, diff_loss_x->stride()},
                        engine,
                        diff_loss_x->data_ptr().get() + diff_loss_x->data_offset());

                    accumulate(
                        diff_loss_x_mem,
                        x->get_grad(),
                        engine,
                        engine_stream);
                }
            }
            
            }
            
            
            {
            if (y->requires_grad())
            {

                auto x_prev_grad = x->requires_grad();
                x->set_requires_grad(false);

                auto diff_loss_y = mul.forward({x, // diff_loss_out_y
                                                diff_loss_out});
                x->set_requires_grad(x_prev_grad);

                if (!y->get_grad())
                {

                    y->set_grad(diff_loss_y);
                }
                else
                {

                    dnnl::memory diff_loss_y_mem(
                        {diff_loss_y->shape(), dnnl::memory::data_type::f32, diff_loss_y->stride()},
                        engine,
                        diff_loss_y->data_ptr().get() + diff_loss_y->data_offset());
                    accumulate(
                        diff_loss_y_mem,
                        y->get_grad(),
                        engine,
                        engine_stream);
                }
            }
            }
        }

    } // ops
} // mt