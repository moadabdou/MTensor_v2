#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <omp.h>
#include <MTensor/tensorImpl.hpp>
#include <MTensor/utils/tensor_iterator.hpp>
#include <MTensor/ops.hpp>

namespace mt
{
    namespace ops
    {

        int64_t Max_reduction::count = 0;

        Max_reduction::Max_reduction(int64_t dim, bool inc_counter) : m_dim(dim)
        {

            if (inc_counter)
            {
                m_name = "Max_reduction" + std::to_string(count);
                count++;
            }
        }

        std::shared_ptr<TensorImpl> Max_reduction::forward(const std::vector<std::shared_ptr<TensorImpl>> &operands)
        {

            auto max_dim = operands[0]->shape().size() - 1;

            if (m_dim > max_dim)
            {
                throw std::invalid_argument("error: Max_reduction() dim is out of range of in_tensor dims");
            }

            auto in_tensor = operands[0];

            if (in_tensor->shape()[m_dim] == 1)
            {
                return in_tensor;
            }

            Contiguous contiguous;
            Transpose transpose(m_dim, max_dim);

            if (m_dim != max_dim)
            { // if not last dim
                in_tensor = contiguous.forward({transpose.forward({in_tensor})});
            }
            else if (!in_tensor->is_contiguous())
            {
                in_tensor = contiguous.forward({in_tensor});
            }

            if (in_tensor->grad_fn())
            {
                in_tensor->grad_fn()->set_operands({}); // free the memory
            }

            const auto &in_shape = in_tensor->shape();
            const auto &in_stride = in_tensor->stride();

            std::vector<int64_t> out_shape = in_shape;
            out_shape[max_dim] = 1;
            std::vector<int64_t> dst_stride = row_major_stride(out_shape);

            std::shared_ptr<float> data_storage = reduce_max_last_dim_avx512(
                in_tensor->data_ptr().get(),
                in_shape,
                in_stride,
                m_max_indices);

            std::shared_ptr<Max_reduction> grad_fn = nullptr;
            bool requires_grad = false;

            if (in_tensor->requires_grad())
            {
                requires_grad = true;
                grad_fn = std::make_shared<Max_reduction>(m_dim, true);
                grad_fn->set_operands({operands[0]});
                grad_fn->m_max_indices = m_max_indices;
            }

            auto out_tensor = std::make_shared<TensorImpl>(data_storage, 0, out_shape, grad_fn, requires_grad, true, dst_stride);

            if (m_dim != max_dim)
            {
                out_tensor = contiguous.forward({transpose.forward({out_tensor})});
                out_tensor->set_grad_fn(grad_fn);
                return out_tensor;
            }

            return out_tensor;
        }

        void Max_reduction::backward(const std::shared_ptr<TensorImpl> &diff_loss_out)
        {
            // if (diff_loss_out->requires_grad()) std::cout << m_name ;
            const auto &x = m_operands[0];

            if (!x->requires_grad())
                return;

            
            {
            if (!x->get_grad())
            {
                x->set_grad(TensorImpl::zeros(x->shape()));
            }
            }

            const auto &x_grad = x->get_grad();
            const auto &x_grad_stride = x_grad->stride();
            const auto &x_grad_data_ptr = x_grad->data_ptr().get() + x_grad->data_offset();
            const auto &out_grad_stride = diff_loss_out->stride();
            const auto &out_grad_data_ptr = diff_loss_out->data_ptr().get() + diff_loss_out->data_offset();

#pragma omp parallel for
            for (int64_t idx = 0; idx < static_cast<int64_t>(m_max_indices.size()); ++idx)
            {
                const auto &el = m_max_indices[idx];

                int64_t offset_x = 0, offset_out = 0;

                for (int64_t i = 0; i < el.first.size(); i++)
                {
                    offset_x += el.first[i] * x_grad_stride[i < m_dim ? i : i + 1];
                    offset_out += el.first[i] * out_grad_stride[i < m_dim ? i : i + 1];
                }
                offset_x += el.second * x_grad_stride[m_dim];


                {
                    *(x_grad_data_ptr + offset_x) += *(out_grad_data_ptr + offset_out);
                }
            }
        }

    } // ops
} // mt