#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <MTensor/tensorImpl.hpp>
#include <MTensor/ops.hpp>

namespace mt
{
    namespace ops
    {

        int64_t RMSNormalization::count = 0;

        RMSNormalization::RMSNormalization(bool inc_counter)
        {

            if (inc_counter)
            {
                m_name = "RMSNormalization" + std::to_string(count);
                count++;
            }
        }

        std::shared_ptr<TensorImpl> RMSNormalization::forward(const std::vector<std::shared_ptr<TensorImpl>> &operands)
        {

            try
            {
                const auto &in_tensor = operands[0];
                const auto &src_shape = in_tensor->shape();
                const auto &scale = operands[1];
                const auto &scale_shape = scale->shape();
                const auto dst_stride = row_major_stride(src_shape);

                if (
                    src_shape.size() < 2 ||
                    scale_shape.size() != 1)
                {
                    throw std::invalid_argument(" in_tensor must be 2D, higher while scale must be 1D contiguous tensors with same shape ");
                }

                if (src_shape.back() != scale_shape[0])
                {
                    throw std::invalid_argument(" the dim of scale must be equal to the last dim of the in_tensor !");
                }

                if (
                    !scale->is_contiguous())
                {
                    throw std::invalid_argument(" scale must be contiguous ");
                }

                dnnl::engine eng(dnnl::engine::kind::cpu, 0);
                dnnl::stream strm(eng);

                std::shared_ptr<float> dst_data(new float[src_shape[0] * dst_stride[0]], std::default_delete<float[]>());

                auto src_md = dnnl::memory::desc(
                    src_shape, dnnl::memory::data_type::f32, in_tensor->stride());
                auto dst_md = dnnl::memory::desc(
                    src_shape, dnnl::memory::data_type::f32, dst_stride);
                auto scale_md = dnnl::memory::desc(
                    scale_shape, dnnl::memory::data_type::f32, dnnl::memory::format_tag::x);

                auto src_mem = dnnl::memory(src_md, eng, in_tensor->data_ptr().get() + in_tensor->data_offset());
                auto dst_mem = dnnl::memory(dst_md, eng, dst_data.get());
                auto scale_mem = dnnl::memory(scale_md, eng, scale->data_ptr().get() + scale->data_offset());

                // Create primitive descriptor.
                m_fwd_rmsnorm_pd = dnnl::layer_normalization_forward::primitive_desc(eng,
                                                                                     dnnl::prop_kind::forward_training, src_md, dst_md, 1.e-10f,
                                                                                     dnnl::normalization_flags::use_scale);

                m_mean = dnnl::memory(m_fwd_rmsnorm_pd.mean_desc(), eng);
                m_variance = dnnl::memory(m_fwd_rmsnorm_pd.variance_desc(), eng);

                // Create the primitive.
                auto rmsnorm_prim = dnnl::layer_normalization_forward(m_fwd_rmsnorm_pd);

                // Primitive arguments. Set up in-place execution by assigning src as DST.
                std::unordered_map<int, dnnl::memory> lnorm_args;
                lnorm_args.insert({DNNL_ARG_SRC, src_mem});
                lnorm_args.insert({DNNL_ARG_MEAN, m_mean});
                lnorm_args.insert({DNNL_ARG_VARIANCE, m_variance});
                lnorm_args.insert({DNNL_ARG_SCALE, scale_mem});
                lnorm_args.insert({DNNL_ARG_DST, dst_mem});
                // Primitive execution: batch normalization with ReLU.
                rmsnorm_prim.execute(strm, lnorm_args);

                // Wait for the computation to finalize.
                strm.wait();

                std::shared_ptr<RMSNormalization> grad_fn = nullptr;
                bool requires_grad = false;

                if (in_tensor->requires_grad() || scale->requires_grad())
                {
                    requires_grad = true;
                    grad_fn = std::make_shared<RMSNormalization>(true);

                    grad_fn->set_operands({in_tensor, scale});
                    grad_fn->m_mean = m_mean;
                    grad_fn->m_variance = m_variance;
                    ;
                    grad_fn->m_fwd_rmsnorm_pd = m_fwd_rmsnorm_pd;
                }

                return std::make_shared<TensorImpl>(dst_data, 0, src_shape, grad_fn, requires_grad, true, dst_stride);
            }
            catch (std::exception &e)
            {

                throw std::invalid_argument(std::string("error: RMSNormalization() was not possible for in_tensor: ") + e.what());
            }
        }

        void RMSNormalization::backward(const std::shared_ptr<TensorImpl> &diff_loss_out)
        {
            // if (diff_loss_out->requires_grad()) std::cout << m_name ;
            dnnl::engine engine(dnnl::engine::kind::cpu, 0);
            dnnl::stream engine_stream(engine);

            const auto &x = m_operands[0];
            const auto &scale = m_operands[1];

            if (x->requires_grad() || scale->requires_grad())
            {

                auto src_md = dnnl::memory::desc(x->shape(), dnnl::memory::data_type::f32, x->stride());
                auto scale_md = dnnl::memory::desc(scale->shape(), dnnl::memory::data_type::f32, scale->stride());
                auto diff_dst_md = dnnl::memory::desc(diff_loss_out->shape(), dnnl::memory::data_type::f32, diff_loss_out->stride());
                auto diff_src_md = dnnl::memory::desc(x->shape(), dnnl::memory::data_type::f32, row_major_stride(x->shape()));
                auto diff_scale_md = dnnl::memory::desc(scale->shape(), dnnl::memory::data_type::f32, dnnl::memory::format_tag::x);

                auto bwd_pd = dnnl::layer_normalization_backward::primitive_desc(
                    engine,
                    dnnl::prop_kind::backward,
                    diff_src_md,
                    diff_dst_md,
                    src_md,
                    1.e-10f,
                    dnnl::normalization_flags::use_scale,
                    m_fwd_rmsnorm_pd);

                dnnl::memory diff_dst_mem(diff_dst_md, engine, diff_loss_out->data_ptr().get() + diff_loss_out->data_offset());
                dnnl::memory src_mem(src_md, engine, x->data_ptr().get() + x->data_offset());
                dnnl::memory scale_mem(scale_md, engine, scale->data_ptr().get() + scale->data_offset());

                dnnl::memory diff_src_mem;
                std::shared_ptr<float> src_data_storage;
                dnnl::memory diff_scale_mem;
                std::shared_ptr<float> scale_data_storage;

                if (x->get_grad())
                {
                    diff_src_mem = dnnl::memory(diff_src_md, engine);
                }
                else
                {
                    src_data_storage = std::shared_ptr<float>(new float[x->numel()], std::default_delete<float[]>());
                    diff_src_mem = dnnl::memory(diff_src_md, engine, src_data_storage.get());
                }

                if (scale->get_grad())
                {
                    diff_scale_mem = dnnl::memory(diff_scale_md, engine);
                }
                else
                {
                    scale_data_storage = std::shared_ptr<float>(new float[scale->numel()], std::default_delete<float[]>());
                    diff_scale_mem = dnnl::memory(diff_scale_md, engine, scale_data_storage.get());
                }

                auto bn_bwd = dnnl::layer_normalization_backward(bwd_pd);
                bn_bwd.execute(engine_stream, {{DNNL_ARG_SRC, src_mem},
                                               {DNNL_ARG_MEAN, m_mean},
                                               {DNNL_ARG_VARIANCE, m_variance},
                                               {DNNL_ARG_SCALE, scale_mem},
                                               {DNNL_ARG_DIFF_DST, diff_dst_mem},

                                               {DNNL_ARG_DIFF_SRC, diff_src_mem},
                                               {DNNL_ARG_DIFF_SCALE, diff_scale_mem}});

                engine_stream.wait();

                
                {
                if (x->get_grad())
                {
                        accumulate(
                            diff_src_mem,
                            x->get_grad(),
                            engine,
                            engine_stream);
                }
                else
                {
                    x->set_grad(std::make_shared<TensorImpl>(src_data_storage, 0, diff_src_md.get_dims(), nullptr, false, true, diff_src_md.get_strides()));
                }
                }

                
                {

                if (scale->get_grad())
                {
                        accumulate(
                            diff_scale_mem,
                            scale->get_grad(),
                            engine,
                            engine_stream);
                }
                else
                {
                    scale->set_grad(std::make_shared<TensorImpl>(scale_data_storage, 0, diff_scale_md.get_dims(), nullptr, false, true, diff_scale_md.get_strides()));
                }

                }
            }
        }

    } // ops
} // mt