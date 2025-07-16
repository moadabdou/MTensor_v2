#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <chrono>

#include <MTensor/tensorImpl.hpp>
#include <MTensor/ops.hpp>
#include <MTensor/utils/braodcast.hpp>

namespace mt
{
    namespace ops
    {

        int64_t Matmul::count = 0;

        Matmul::Matmul(bool inc_counter)
        {
            if (inc_counter)
            {
                m_name = "Matmul" + std::to_string(count);
                count++;
            }
        }

        std::shared_ptr<TensorImpl> Matmul::forward(const std::vector<std::shared_ptr<TensorImpl>> &operands)
        {

            try
            {

                if (operands.size() != 2 && operands.size() != 3)
                {
                    throw std::invalid_argument(" invalide operand number be 2 or 3");
                }

                auto [in_tensor_0, in_tensor_1] = utils::broadcast_matmul({operands[0],
                                                                           operands[1]});

                const auto &in_tensor_0_shape = in_tensor_0->shape();
                const auto &in_tensor_1_shape = in_tensor_1->shape();
                const auto &in_tensor_0_stride = in_tensor_0->stride();
                const auto &in_tensor_1_stride = in_tensor_1->stride();

                auto dst_shape = in_tensor_0_shape;
                dst_shape[dst_shape.size() - 1] = in_tensor_1_shape[in_tensor_1_shape.size() - 1];

                if (operands.size() != 2 && operands[2]->shape().size() != 1)
                {
                    throw std::invalid_argument(" bias must be a 1D tensor");
                }

                std::vector<int64_t> view_shape(dst_shape.size(), 1ll);
                view_shape.back() = dst_shape.back();

                const auto &in_tensor_2 = operands.size() != 2 ? ops::View(view_shape).forward({operands[2]}) : nullptr;

                const auto &dst_stride = mt::row_major_stride(dst_shape);
                const auto &dst_numel = dst_stride[0] * dst_shape[0];

                std::shared_ptr<float> dst_data(new float[dst_numel], std::default_delete<float[]>());

                dnnl::engine engine(dnnl::engine::kind::cpu, 0);
                dnnl::stream strm(engine);

                auto in_tensor_0_md = dnnl::memory::desc(in_tensor_0_shape, dnnl::memory::data_type::f32, in_tensor_0_stride);
                auto in_tensor_1_md = dnnl::memory::desc(in_tensor_1_shape, dnnl::memory::data_type::f32, in_tensor_1_stride);
                auto dst_md = dnnl::memory::desc(dst_shape, dnnl::memory::data_type::f32, dst_stride);

                auto in_tensor_0_mem = dnnl::memory(in_tensor_0_md, engine, in_tensor_0->data_ptr().get() + in_tensor_0->data_offset());
                auto in_tensor_1_mem = dnnl::memory(in_tensor_1_md, engine, in_tensor_1->data_ptr().get() + in_tensor_1->data_offset());
                auto dst_mem = dnnl::memory(dst_md, engine, dst_data.get());

                std::unordered_map<int, dnnl::memory> matmul_args;

                matmul_args.insert({DNNL_ARG_SRC, in_tensor_0_mem});
                matmul_args.insert({DNNL_ARG_WEIGHTS, in_tensor_1_mem});
                matmul_args.insert({DNNL_ARG_DST, dst_mem});

                if (in_tensor_2)
                {

                    auto in_tensor_2_md = dnnl::memory::desc(in_tensor_2->shape(), dnnl::memory::data_type::f32, in_tensor_2->stride());
                    auto in_tensor_2_mem = dnnl::memory(in_tensor_2_md, engine, in_tensor_2->data_ptr().get() + in_tensor_2->data_offset());

                    matmul_args.insert({DNNL_ARG_BIAS, in_tensor_2_mem});

                    auto matmul_pd = dnnl::matmul::primitive_desc(engine, in_tensor_0_md, in_tensor_1_md, in_tensor_2_md, dst_md);

                    auto matmul_prim = dnnl::matmul(matmul_pd);

                    matmul_prim.execute(strm, matmul_args);
                }
                else
                {

                    auto matmul_pd = dnnl::matmul::primitive_desc(engine, in_tensor_0_md, in_tensor_1_md, dst_md);

                    auto matmul_prim = dnnl::matmul(matmul_pd);

                    matmul_prim.execute(strm, matmul_args);
                }

                strm.wait();

                std::shared_ptr<Operation> grad_fn = nullptr;
                bool requires_grad = false;

                if (
                    in_tensor_0->requires_grad() ||
                    in_tensor_1->requires_grad() ||
                    in_tensor_2 && in_tensor_2->requires_grad())
                {
                    requires_grad = true;
                    grad_fn = std::make_shared<Matmul>(true);
                    if (in_tensor_2)
                    {
                        grad_fn->set_operands({in_tensor_0, in_tensor_1, in_tensor_2});
                    }
                    else
                    {
                        grad_fn->set_operands({in_tensor_0, in_tensor_1});
                    }
                }

                return std::make_shared<TensorImpl>(dst_data, 0, dst_shape, grad_fn, requires_grad, true, dst_stride);
            }
            catch (std::exception &e)
            {
                throw std::runtime_error(
                    std::string("error : Matmul() ") + e.what());
            }
        }

        void Matmul::backward(const std::shared_ptr<TensorImpl> &diff_loss_out)
        {
            // if (diff_loss_out->requires_grad()) std::cout << m_name ;

            dnnl::engine engine(dnnl::engine::kind::cpu, 0);
            dnnl::stream engine_stream(engine);

            auto &x = m_operands[0];
            auto &w = m_operands[1];
            const auto &dims_x = x->shape().size() - 1;

            Transpose transpose(dims_x, dims_x - 1); // w,x are broadcasted before matmul

            Matmul matmul;

            
            {
            if (x->requires_grad())
            {

                bool prev_state = w->requires_grad();
                w->set_requires_grad(false);

                auto diff_loss_x = matmul.forward({diff_loss_out, transpose.forward({w})});

                w->set_requires_grad(prev_state);

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
            if (w->requires_grad())
            {

                bool prev_state = x->requires_grad();
                x->set_requires_grad(false);

                auto diff_loss_w = matmul.forward({transpose.forward({x}), diff_loss_out});

                x->set_requires_grad(prev_state);

                if (!w->get_grad())
                {
                    w->set_grad(diff_loss_w);
                }
                else
                {

                    dnnl::memory diff_loss_w_mem(
                        {diff_loss_w->shape(), dnnl::memory::data_type::f32, diff_loss_w->stride()},
                        engine,
                        diff_loss_w->data_ptr().get() + diff_loss_w->data_offset());
                        accumulate(
                            diff_loss_w_mem,
                            w->get_grad(),
                            engine,
                            engine_stream);
                }
            }

            }
            
            {
            if (m_operands.size() > 2 && m_operands[2]->requires_grad())
            {

                const auto &b = m_operands[2];
                std::vector<int64_t> sum_dims(diff_loss_out->shape().size() - 1);
                std::iota(sum_dims.begin(), sum_dims.end(), 0);

                Sum sum(sum_dims);

                auto diff_loss_b = sum.forward({diff_loss_out});

                if (!b->get_grad())
                {
                    b->set_grad(diff_loss_b);
                }
                else
                {

                    dnnl::memory diff_loss_b_mem(
                        {diff_loss_b->shape(), dnnl::memory::data_type::f32, diff_loss_b->stride()},
                        engine,
                        diff_loss_b->data_ptr().get() + diff_loss_b->data_offset());
                        accumulate(
                            diff_loss_b_mem,
                            b->get_grad(),
                            engine,
                            engine_stream);
                }
            }
            
            }
        }

    } // ops
} // mt