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

        int64_t Sub::count = 0;

        Sub::Sub(bool inc_counter)
        {
            if (inc_counter)
            {
                m_name = "Sub" + std::to_string(count);
                count++;
            }
        }

        std::shared_ptr<TensorImpl> Sub::forward(const std::vector<std::shared_ptr<TensorImpl>> &operands)
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
                    dnnl::algorithm::binary_sub,
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
                    grad_fn = std::make_shared<Sub>(true);
                    grad_fn->set_operands({in_tensor_0, in_tensor_1});
                }

                return std::make_shared<TensorImpl>(data_storage, 0, in_shape, grad_fn, requires_grad, true, dst_stride);
            }
            catch (std::exception &e)
            {
                throw std::runtime_error(
                    std::string("error : Sub() ") + e.what());
            }
        }

        void Sub::backward(const std::shared_ptr<TensorImpl> &diff_loss_out)
        {
            dnnl::engine engine(dnnl::engine::kind::cpu, 0);
            dnnl::stream engine_stream(engine);

            Linear linear(-1, 0);

            auto &x = m_operands[0];
            auto &y = m_operands[1];

            const auto &diff_loss_x = diff_loss_out;

            dnnl::memory::desc diff_x_md({diff_loss_x->shape(), dnnl::memory::data_type::f32, diff_loss_x->stride()});

            dnnl::memory diff_loss_x_mem(
                diff_x_md,
                engine,
                diff_loss_x->data_ptr().get() + diff_loss_x->data_offset());
           
            
            {
            if (x->requires_grad())
            {

                if (!x->get_grad())
                {

                    std::shared_ptr<float> data_storage(new float[diff_loss_out->numel()], std::default_delete<float[]>());

                    dnnl::memory dst_m(diff_x_md, engine, data_storage.get());

                    dnnl::reorder reorder_prm(diff_loss_x_mem, dst_m);
                    reorder_prm.execute(engine_stream, diff_loss_x_mem, dst_m);

                    engine_stream.wait();

                    x->set_grad(std::make_shared<TensorImpl>(data_storage, 0, diff_x_md.get_dims(), nullptr, false, true, diff_x_md.get_strides()));
                }
                else
                {

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

                y->set_requires_grad(false);

                auto diff_loss_y = linear.forward({diff_loss_out});

                y->set_requires_grad(true);

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