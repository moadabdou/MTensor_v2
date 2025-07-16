#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <MTensor/tensorImpl.hpp>
#include <MTensor/ops.hpp>

namespace mt
{
    namespace ops
    {

        int64_t Deconv1d::count = 0;

        Deconv1d::Deconv1d(
            const std::vector<int64_t> &strides,
            const std::vector<int64_t> &padding_l,
            const std::vector<int64_t> &padding_r,
            bool inc_counter)
        {

            if (strides.size() != 1 || strides[0] < 0)
            {
                throw std::invalid_argument("error : Deconv1d() invalid strides were given");
            }
            if (padding_l.size() != 1 || padding_l[0] < 0)
            {
                throw std::invalid_argument("error : Deconv1d() invalid padding_l was given");
            }
            if (padding_r.size() != 1 || padding_r[0] < 0)
            {
                throw std::invalid_argument("error : Deconv1d() invalid padding_r was given");
            }

            m_strides = strides;
            m_padding_l = padding_l;
            m_padding_r = padding_r;

            if (inc_counter)
            {
                m_name = "Deconv1d" + std::to_string(count);
                count++;
            }
        }

        std::shared_ptr<TensorImpl> Deconv1d::forward(const std::vector<std::shared_ptr<TensorImpl>> &operands)
        {

            try
            {
                const auto &in_tensor = operands[0];
                const auto &src_shape = in_tensor->shape();
                const auto &weights = operands[1];
                const auto &weights_shape = weights->shape();
                const auto &bias = operands.size() > 2 ? operands[2] : nullptr;

                if (
                    src_shape.size() != 3 ||
                    weights_shape.size() != 3 ||
                    bias && bias->shape().size() != 1)
                {
                    throw std::invalid_argument(" in_tensor and weights must be of shape (B,C,T) while bias must be 1d tensor ");
                }

                if (
                    src_shape[1] != weights_shape[1])
                {
                    throw std::invalid_argument(" in_tensor and weights must have same channels number (if in_tensor is (B,C,W) then weights must be (OC,C,KW)) ! ");
                }

                if (
                    bias && bias->shape()[0] != weights_shape[0])
                {
                    throw std::invalid_argument(" bias and weights must have same out_channels number (if bias is (OC) then weights must be (OC,C,KW)) ! ");
                }

                dnnl::engine eng(dnnl::engine::kind::cpu, 0);
                dnnl::stream strm(eng);

                const auto OW = (src_shape[2] - 1) * m_strides[0] - m_padding_l[0] - m_padding_r[0] + weights_shape[2];

                dnnl::memory::dims dst_dims = {src_shape[0], weights_shape[0], OW};

                const auto dst_strides = row_major_stride(dst_dims);

                auto dst_data = custom_deconv_op_forward(
                    in_tensor,
                    weights,
                    bias,
                    dst_dims,
                    dst_strides,
                    m_strides,
                    m_padding_l,
                    m_padding_r,
                    eng,
                    strm,
                    m_fwd_deconv_pd);

                std::shared_ptr<Deconv1d> grad_fn = nullptr;
                bool requires_grad = false;

                if (in_tensor->requires_grad() || weights->requires_grad() || bias->requires_grad())
                {
                    requires_grad = true;
                    grad_fn = std::make_shared<Deconv1d>(
                        m_strides,
                        m_padding_l,
                        m_padding_r,
                        true);
                    grad_fn->m_fwd_deconv_pd = m_fwd_deconv_pd;
                    if (bias)
                        grad_fn->set_operands({in_tensor, weights, bias});
                    else
                        grad_fn->set_operands({in_tensor, weights});
                }

                return std::make_shared<TensorImpl>(dst_data, 0, dst_dims, grad_fn, requires_grad, true, dst_strides);
            }
            catch (std::exception &e)
            {

                throw std::invalid_argument(std::string("error: Deconv1d() was not possible for in_tensor: ") + e.what());
            }
        }

        void Deconv1d::backward(const std::shared_ptr<TensorImpl> &diff_loss_out)
        {
            dnnl::engine engine(dnnl::engine::kind::cpu, 0);
            dnnl::stream engine_stream(engine);

            const auto &x = m_operands[0];
            const auto &w = m_operands[1];
            const auto &b = m_operands.size() > 2 ? m_operands[2] : nullptr;

            deconv_backward(
                engine,
                engine_stream,
                x,w,b,
                diff_loss_out,
                m_fwd_deconv_pd,
                m_strides,
                m_padding_l,
                m_padding_r
            );
        }

    } // ops
} // mt