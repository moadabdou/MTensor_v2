#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <MTensor/tensorImpl.hpp>
#include <MTensor/ops.hpp>

namespace mt
{
    namespace ops
    {

        int64_t Conv2d::count = 0;

        Conv2d::Conv2d(
            const std::vector<int64_t> &strides,
            const std::vector<int64_t> &padding_l,
            const std::vector<int64_t> &padding_r,

            bool inc_counter)
        {

            if (strides.size() != 2 || strides[0] < 0 || strides[1] < 0)
            {
                throw std::invalid_argument("error : Conv2d() invalid strides were given");
            }
            if (padding_l.size() != 2 || padding_l[0] < 0 || padding_l[1] < 0)
            {
                throw std::invalid_argument("error : Conv2d() invalid padding_l was given");
            }
            if (padding_r.size() != 2 || padding_r[0] < 0 || padding_r[1] < 0)
            {
                throw std::invalid_argument("error : Conv2d() invalid padding_r was given");
            }

            m_strides = strides;
            m_padding_l = padding_l;
            m_padding_r = padding_r;

            if (inc_counter)
            {
                m_name = "Conv2d" + std::to_string(count);
                count++;
            }
        }

        std::shared_ptr<TensorImpl> Conv2d::forward(const std::vector<std::shared_ptr<TensorImpl>> &operands)
        {

            try
            {
                const auto &in_tensor = operands[0];
                const auto &src_shape = in_tensor->shape();
                const auto &weights = operands[1];
                const auto &weights_shape = weights->shape();
                const auto &bias = operands.size() > 2 ? operands[2] : nullptr;

                if (
                    src_shape.size() != 4 ||
                    weights_shape.size() != 4 ||
                    bias && bias->shape().size() != 1)
                {
                    throw std::invalid_argument(" in_tensor and weights must be of shape (B,C,H,W) while bias must be 1d tensor ");
                }

                if (
                    src_shape[1] != weights_shape[1])
                {
                    throw std::invalid_argument(" in_tensor and weights must have same channels number (if in_tensor is (B,C,H,W) then weights must be (OC,C,KH,KW)) ! ");
                }

                if (
                    bias && bias->shape()[0] != weights_shape[0])
                {
                    throw std::invalid_argument(" bias and weights must have same out_channels number (if bias is (OC) then weights must be (OC,C,KH,KW)) ! ");
                }

                dnnl::engine eng(dnnl::engine::kind::cpu, 0);
                dnnl::stream strm(eng);

                const auto OH = (src_shape[2] - weights_shape[2] + m_padding_l[0] + m_padding_r[0]) / m_strides[0] + 1;
                const auto OW = (src_shape[3] - weights_shape[3] + m_padding_l[1] + m_padding_r[1]) / m_strides[1] + 1;

                dnnl::memory::dims dst_dims = {src_shape[0], weights_shape[0], OH, OW};

                const auto dst_strides = row_major_stride(dst_dims);

                auto dst_data = custom_conv_op_forward(
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
                    m_conv_fwd_pd);

                std::shared_ptr<Conv2d> grad_fn = nullptr;
                bool requires_grad = false;

                if (in_tensor->requires_grad() || weights->requires_grad() || bias->requires_grad())
                {
                    requires_grad = true;
                    grad_fn = std::make_shared<Conv2d>(
                        m_strides,
                        m_padding_l,
                        m_padding_r,
                        true);
                    grad_fn->m_conv_fwd_pd = m_conv_fwd_pd;
                    if (bias)
                        grad_fn->set_operands({in_tensor, weights, bias});
                    else
                        grad_fn->set_operands({in_tensor, weights});
                }

                return std::make_shared<TensorImpl>(dst_data, 0, dst_dims, grad_fn, requires_grad, true, dst_strides);
            }
            catch (std::exception &e)
            {

                throw std::invalid_argument(std::string("error: Conv2d() was not possible for in_tensor: ") + e.what());
            }
        }

        void Conv2d::backward(const std::shared_ptr<TensorImpl> &diff_loss_out)
        {
            dnnl::engine engine(dnnl::engine::kind::cpu, 0);
            dnnl::stream engine_stream(engine);

            const auto &x = m_operands[0];
            const auto &w = m_operands[1];
            const auto &b = m_operands.size() > 2 ? m_operands[2] : nullptr;

            conv_backward(
                x, w, b,
                diff_loss_out,
                m_conv_fwd_pd,
                m_strides,
                m_padding_l,
                m_padding_r,
                engine,
                engine_stream
            );
        
        }

    } // ops
} // mt