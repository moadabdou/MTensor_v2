#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <MTensor/tensorImpl.hpp>
#include <MTensor/ops.hpp>

namespace mt
{
    namespace ops
    {

        int64_t SoftmaxLog::count = 0;

        SoftmaxLog::SoftmaxLog(int dim, bool inc_counter)
        {
            if (dim < 0)
            {
                throw std::invalid_argument("error : SoftmaxLog() invalid dim was given");
            }

            m_dim = dim;

            if (inc_counter)
            {
                m_name = "SoftmaxLog" + std::to_string(count);
                count++;
            }
        }

        std::shared_ptr<TensorImpl> SoftmaxLog::forward(const std::vector<std::shared_ptr<TensorImpl>> &operands)
        {

            const auto &in_tensor = operands[0];

            const auto &in_shape = in_tensor->shape();
            const auto &src_stride = in_tensor->stride();

            if (m_dim >= in_shape.size())
            {
                throw std::invalid_argument("error: SoftmaxLog() dim is out of range of in_tensor dims");
            }

            const auto &dst_stride = row_major_stride(in_shape);

            std::shared_ptr<float> dst_data(new float[dst_stride[0] * in_shape[0]], std::default_delete<float[]>());

            dnnl::engine eng(dnnl::engine::kind::cpu, 0);
            dnnl::stream strm(eng);

            dnnl::memory::desc src_md(in_shape, dnnl::memory::data_type::f32, src_stride);
            dnnl::memory::desc dst_md(in_shape, dnnl::memory::data_type::f32, dst_stride);

            dnnl::memory src_m(src_md, eng, in_tensor->data_ptr().get() + in_tensor->data_offset());
            dnnl::memory dst_m(dst_md, eng, dst_data.get());

            dnnl::softmax_forward::primitive_desc softmaxLog_desc(
                eng,
                dnnl::prop_kind::forward_inference,
                dnnl::algorithm::softmax_log,
                src_md,
                dst_md,
                m_dim);

            auto softmaxLog = dnnl::softmax_forward(softmaxLog_desc);

            std::unordered_map<int, dnnl::memory> softmaxLog_args;
            softmaxLog_args.insert({DNNL_ARG_SRC, src_m});
            softmaxLog_args.insert({DNNL_ARG_DST, dst_m});

            softmaxLog.execute(strm, softmaxLog_args);

            strm.wait();

            std::shared_ptr<SoftmaxLog> grad_fn = nullptr;
            bool requires_grad = false;

            if (in_tensor->requires_grad())
            {
                requires_grad = true;
                grad_fn = std::make_shared<SoftmaxLog>(m_dim, true);
                grad_fn->set_operands({in_tensor});
            }

            grad_fn->m_dst_tensor = std::make_shared<TensorImpl>(dst_data, 0, in_shape, grad_fn, requires_grad, true, dst_stride);
            return grad_fn->m_dst_tensor;
        }

        void SoftmaxLog::backward(const std::shared_ptr<TensorImpl> &diff_loss_out)
        {
            // if (diff_loss_out->requires_grad()) std::cout << m_name ;
            dnnl::engine engine(dnnl::engine::kind::cpu, 0);
            dnnl::stream engine_stream(engine);

            const auto &x = m_operands[0];
            if (!x->requires_grad())
                return;

            auto src_md = dnnl::memory::desc(x->shape(), dnnl::memory::data_type::f32, x->stride());
            auto diff_src_md = dnnl::memory::desc(x->shape(), dnnl::memory::data_type::f32, row_major_stride(x->shape()));
            auto dst_md = dnnl::memory::desc(m_dst_tensor->shape(), dnnl::memory::data_type::f32, m_dst_tensor->stride());
            auto diff_dst_md = dnnl::memory::desc(diff_loss_out->shape(), dnnl::memory::data_type::f32, diff_loss_out->stride());

            dnnl::softmax_forward::primitive_desc hint_fwd_pd(
                engine,
                dnnl::prop_kind::forward_inference,
                dnnl::algorithm::softmax_log,
                src_md,
                dst_md,
                m_dim);

            // Backward primitive desc
            auto bwd_pd = dnnl::softmax_backward::primitive_desc(
                engine,
                dnnl::algorithm::softmax_log,
                diff_src_md,
                diff_dst_md,
                dst_md,
                m_dim,
                hint_fwd_pd);

            dnnl::memory dst_mem(dst_md, engine, m_dst_tensor->data_ptr().get() + m_dst_tensor->data_offset());
            dnnl::memory diff_dst_mem(diff_dst_md, engine, diff_loss_out->data_ptr().get() + diff_loss_out->data_offset());
            dnnl::memory diff_src_mem;
            std::shared_ptr<float> data_storage; // in case if the x_grad does not exist

            if (x->get_grad())
            {
                diff_src_mem = dnnl::memory(diff_src_md, engine); // x_grad exists so we allocate new temporary memory
            }
            else
            {
                data_storage = std::shared_ptr<float>(new float[x->numel()], std::default_delete<float[]>());
                diff_src_mem = dnnl::memory(diff_src_md, engine, data_storage.get()); // x_grad does not exists so we make one and the result will directly routed to it
            }

            auto softmax_bwd = dnnl::softmax_backward(bwd_pd);
            softmax_bwd.execute(engine_stream, {{DNNL_ARG_DST, dst_mem},
                                                {DNNL_ARG_DIFF_DST, diff_dst_mem},
                                                {DNNL_ARG_DIFF_SRC, diff_src_mem}});

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
                x->set_grad(std::make_shared<TensorImpl>(data_storage, 0, diff_src_md.get_dims(), nullptr, false, true, diff_src_md.get_strides()));
            }
            }
        }

    } // ops
} // mt