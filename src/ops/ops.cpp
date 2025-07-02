#include <unordered_map>
#include <stdexcept>
#include <MTensor/tensorImpl.hpp>
#include <MTensor/ops.hpp>

namespace mt{

namespace ops{


void accumulate(
    dnnl::memory& src,
    const std::shared_ptr<TensorImpl>& dst,
    dnnl::engine engine,
    dnnl::stream stream_engine
){

    auto dst_md = dnnl::memory::desc(
            dst->shape() , dnnl::memory::data_type::f32, dst->stride());

    auto dst_mem = dnnl::memory(dst_md, engine, dst->data_ptr().get() + dst->data_offset() );

    auto binary_pd = dnnl::binary::primitive_desc(engine, dnnl::algorithm::binary_add,
            src.get_desc() , dst_md, dst_md);

    auto binary_prim = dnnl::binary(binary_pd);

    std::unordered_map<int, dnnl::memory> binary_args;
    binary_args.insert({DNNL_ARG_SRC_0, src});
    binary_args.insert({DNNL_ARG_SRC_1, dst_mem});
    binary_args.insert({DNNL_ARG_DST, dst_mem});

    binary_prim.execute(stream_engine, binary_args);

    stream_engine.wait();

}

std::shared_ptr<float> custom_eltwise_op(
    const std::shared_ptr<TensorImpl>& in_tensor, 
    const dnnl::eltwise_forward::primitive_desc& primitive_desc, 
    const dnnl::engine& engine,
    const dnnl::memory::desc& src_md,
    const dnnl::memory::desc& dst_md
){
    
    dnnl::stream strm(engine);

    std::shared_ptr<float> dst_mp(new float[in_tensor->numel()], std::default_delete<float[]>());

    dnnl::memory src_m(src_md, engine, in_tensor->data_ptr().get() + in_tensor->data_offset());
    dnnl::memory dst_m(dst_md, engine, dst_mp.get());

    dnnl::eltwise_forward eltwise_op(primitive_desc);
    
    std::unordered_map<int, dnnl::memory> op_args;

    op_args.insert({DNNL_ARG_SRC, src_m});
    op_args.insert({DNNL_ARG_DST, dst_m});

    eltwise_op.execute(strm, op_args);

    strm.wait();

    return dst_mp;
}


//input tensor are of the same shape 
std::shared_ptr<float> custom_binary_op(
    const std::shared_ptr<TensorImpl>& in_tensor_0, 
    const std::shared_ptr<TensorImpl>& in_tensor_1, 
    const dnnl::binary::primitive_desc& primitive_desc, 
    const dnnl::engine& engine,
    const dnnl::memory::desc& src_0_md,
    const dnnl::memory::desc& src_1_md,
    const dnnl::memory::desc& dst_md
){

    dnnl::stream strm(engine);

    //prepare memories 

    const auto& dst_mp = std::shared_ptr<float>(new float[in_tensor_0->numel()], std::default_delete<float[]>());

    const auto& in_tensor_0_mem = dnnl::memory(src_0_md, engine, in_tensor_0->data_ptr().get() + in_tensor_0->data_offset());
    const auto& in_tensor_1_mem = dnnl::memory(src_1_md, engine, in_tensor_1->data_ptr().get() + in_tensor_1->data_offset());
    const auto& dst_mem = dnnl::memory(dst_md, engine, dst_mp.get());


    //binary execution
    dnnl::binary binary_op(primitive_desc);

    std::unordered_map<int, dnnl::memory> op_args;

    op_args.insert({DNNL_ARG_SRC_0, in_tensor_0_mem});
    op_args.insert({DNNL_ARG_SRC_1, in_tensor_1_mem});
    op_args.insert({DNNL_ARG_DST, dst_mem});

    binary_op.execute(strm, op_args);

    strm.wait();

    return dst_mp;
}


std::shared_ptr<float> custom_reduction_op(
    const std::shared_ptr<TensorImpl>& in_tensor, 
    const dnnl::reduction::primitive_desc& primitive_desc, 
    const dnnl::engine& engine,
    const dnnl::memory::desc& src_md,
    const dnnl::memory::desc& dst_md
){
    
    dnnl::stream strm(engine);

    std::shared_ptr<float> dst_mp(new float[ dst_md.get_dims()[0] * dst_md.get_strides()[0] ], std::default_delete<float[]>());

    dnnl::memory src_m(src_md, engine, in_tensor->data_ptr().get() + in_tensor->data_offset());
    dnnl::memory dst_m(dst_md, engine, dst_mp.get());

    dnnl::reduction reduction_op(primitive_desc);
    
    std::unordered_map<int, dnnl::memory> op_args;

    op_args.insert({DNNL_ARG_SRC, src_m});
    op_args.insert({DNNL_ARG_DST, dst_m});

    reduction_op.execute(strm, op_args);

    strm.wait();

    return dst_mp;
}


//pooling  

std::shared_ptr<float> custom_pooling_op_forward(
    const std::shared_ptr<TensorImpl>& in_tensor,
    dnnl::algorithm pool_algorithm, 
    const dnnl::memory::dims& dst_dims,    
    const dnnl::memory::dims& kernel,
    const dnnl::memory::dims& strides,
    const dnnl::memory::dims& padding_l,
    const dnnl::memory::dims& padding_r,
    dnnl::engine& engine,
    dnnl::stream& stream,
    dnnl::pooling_forward::primitive_desc& pool_fwd_pd,
    bool need_work_space,
    std::unique_ptr<dnnl::memory>& workspace_mem
){

    
    auto src_md = dnnl::memory::desc(in_tensor->shape() , dnnl::memory::data_type::f32, in_tensor->stride());
    // Let oneDNN determine the best output dimensions and format
    auto dst_md = dnnl::memory::desc(dst_dims , dnnl::memory::data_type::f32, dnnl::memory::format_tag::any);

    pool_fwd_pd = dnnl::pooling_forward::primitive_desc(
        engine,
        need_work_space ? dnnl::prop_kind::forward_training  : dnnl::prop_kind::forward_inference, 
        pool_algorithm,
        src_md,
        dst_md,
        strides,
        kernel,
        {0,0,0},
        padding_l, 
        padding_r
    );

    const auto& dst_shape = pool_fwd_pd.dst_desc().get_dims();
    const auto& dst_stride = row_major_stride(dst_shape);

    std::shared_ptr<float> dst_data(new float[dst_shape[0] * dst_stride[0]], std::default_delete<float[]>());

    auto src_m = dnnl::memory(src_md , engine , in_tensor->data_ptr().get() + in_tensor->data_offset());
    auto dst_m = dnnl::memory(pool_fwd_pd.dst_desc(), engine, dst_data.get());

    auto pooling = dnnl::pooling_forward(pool_fwd_pd);

    std::unordered_map<int, dnnl::memory> op_args;

    op_args.insert({DNNL_ARG_SRC, src_m});
    op_args.insert({DNNL_ARG_DST, dst_m});

    if (need_work_space){
        workspace_mem = std::make_unique<dnnl::memory>(pool_fwd_pd.workspace_desc(), engine);
        op_args.insert({DNNL_ARG_WORKSPACE, *workspace_mem});
    }

    pooling.execute(stream, op_args);
    stream.wait();

    return dst_data;
}



std::shared_ptr<float> custom_conv_op_forward(
    const std::shared_ptr<TensorImpl>& in_tensor,
    const std::shared_ptr<TensorImpl>& weights,
    const std::shared_ptr<TensorImpl>& bias,
    const dnnl::memory::dims& dst_dims,
    const dnnl::memory::dims& dst_strides,  
    const dnnl::memory::dims& strides,
    const dnnl::memory::dims& padding_l,
    const dnnl::memory::dims& padding_r,
    dnnl::engine& engine,
    dnnl::stream& stream
){

    const auto& src_dims = in_tensor->shape();
    const auto& src_stride = in_tensor->stride();
    const auto& weights_dims = weights->shape();
    const auto& weights_stride = weights->stride();


    auto user_src_mem = dnnl::memory(
            {src_dims, dnnl::memory::data_type::f32, src_stride},
            engine, in_tensor->data_ptr().get() + in_tensor->data_offset());
    auto user_weights_mem = dnnl::memory(
            {weights_dims , dnnl::memory::data_type::f32, weights_stride},
            engine, weights->data_ptr().get() + weights->data_offset());
    
    const auto dst_numel = dst_strides[0] * dst_dims[0];


    std::shared_ptr<float> dst_data(new float[dst_numel], std::default_delete<float[]>());
    
    auto user_dst_mem = dnnl::memory(
            {dst_dims, dnnl::memory::data_type::f32, dst_strides},
            engine, dst_data.get());

    auto conv_src_md = dnnl::memory::desc(
            src_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::any);
    auto conv_weights_md = dnnl::memory::desc(
            weights_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::any);
    auto conv_dst_md = dnnl::memory::desc(
            dst_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::any);
    
    
    
    auto user_bias_md = bias
            ? dnnl::memory::desc( bias->shape(), dnnl::memory::data_type::f32, bias->stride())
            : dnnl::memory::desc();
    auto user_bias_mem = bias ? dnnl::memory(user_bias_md, engine, bias->data_ptr().get() + bias->data_offset()) : dnnl::memory(user_bias_md, engine);

    auto conv_pd = dnnl::convolution_forward::primitive_desc(engine,
            dnnl::prop_kind::forward_training, dnnl::algorithm::convolution_direct,
            conv_src_md, conv_weights_md, user_bias_md, conv_dst_md,
            strides , padding_l, padding_r);
        
    auto conv_src_mem = user_src_mem;
    auto conv_weights_mem = user_weights_mem;
    auto conv_dst_mem = user_dst_mem;

    if (conv_pd.src_desc() != user_src_mem.get_desc()) {
        conv_src_mem = dnnl::memory(conv_pd.src_desc(), engine);
        dnnl::reorder(user_src_mem, conv_src_mem)
                .execute(stream, user_src_mem, conv_src_mem);
    }

    if (conv_pd.weights_desc() != user_weights_mem.get_desc()) {
        conv_weights_mem = dnnl::memory(conv_pd.weights_desc(), engine);
        dnnl::reorder(user_weights_mem, conv_weights_mem)
                .execute(stream, user_weights_mem, conv_weights_mem);
    }

    if (conv_pd.dst_desc() != user_dst_mem.get_desc()) {
        conv_dst_mem = dnnl::memory(conv_pd.dst_desc(), engine);
    }

    auto conv_prim = dnnl::convolution_forward(conv_pd);

    std::unordered_map<int, dnnl::memory> conv_args;
    conv_args.insert({DNNL_ARG_SRC, conv_src_mem});
    conv_args.insert({DNNL_ARG_WEIGHTS, conv_weights_mem});
    conv_args.insert({DNNL_ARG_BIAS, user_bias_mem});
    conv_args.insert({DNNL_ARG_DST, conv_dst_mem});

    conv_prim.execute(stream, conv_args);

    if (conv_pd.dst_desc() != user_dst_mem.get_desc()) {
        dnnl::reorder(conv_dst_mem, user_dst_mem)
                .execute(stream, conv_dst_mem, user_dst_mem);
    } else
        user_dst_mem = conv_dst_mem;
    
    stream.wait();
    
    return dst_data;
}


std::shared_ptr<float> custom_deconv_op_forward(
    const std::shared_ptr<TensorImpl>& in_tensor,
    const std::shared_ptr<TensorImpl>& weights,
    const std::shared_ptr<TensorImpl>& bias,
    const dnnl::memory::dims& dst_dims,
    const dnnl::memory::dims& dst_strides,    
    const dnnl::memory::dims& strides,
    const dnnl::memory::dims& padding_l,
    const dnnl::memory::dims& padding_r,
    dnnl::engine& engine,
    dnnl::stream& stream
){
    const auto& src_dims = in_tensor->shape();
    const auto& src_stride = in_tensor->stride();
    const auto& weights_dims = weights->shape();
    const auto& weights_stride = weights->stride();


    auto user_src_mem = dnnl::memory(
            {src_dims, dnnl::memory::data_type::f32, src_stride},
            engine, in_tensor->data_ptr().get() + in_tensor->data_offset());
    auto user_weights_mem = dnnl::memory(
            {weights_dims , dnnl::memory::data_type::f32, weights_stride},
            engine, weights->data_ptr().get() + weights->data_offset());
    
    const auto dst_numel = dst_strides[0] * dst_dims[0];


    std::shared_ptr<float> dst_data(new float[dst_numel], std::default_delete<float[]>());
    
    auto user_dst_mem = dnnl::memory(
            {dst_dims, dnnl::memory::data_type::f32, dst_strides},
            engine, dst_data.get());

    auto deconv_src_md = dnnl::memory::desc(
            src_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::any);
    auto deconv_weights_md = dnnl::memory::desc(
            weights_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::any);
    auto deconv_dst_md = dnnl::memory::desc(
            dst_dims, dnnl::memory::data_type::f32, dnnl::memory::format_tag::any);
    
    
    
    auto user_bias_md = bias
            ? dnnl::memory::desc( bias->shape(), dnnl::memory::data_type::f32, bias->stride())
            : dnnl::memory::desc();
    auto user_bias_mem = bias ? dnnl::memory(user_bias_md, engine, bias->data_ptr().get() + bias->data_offset()) : dnnl::memory(user_bias_md, engine);

    auto deconv_pd = dnnl::deconvolution_forward::primitive_desc(engine,
            dnnl::prop_kind::forward_training, dnnl::algorithm::deconvolution_direct,
            deconv_src_md, deconv_weights_md, user_bias_md, deconv_dst_md,
            strides , padding_l, padding_r);
        
    auto deconv_src_mem = user_src_mem;
    auto deconv_weights_mem = user_weights_mem;
    auto deconv_dst_mem = user_dst_mem;

    if (deconv_pd.src_desc() != user_src_mem.get_desc()) {
        deconv_src_mem = dnnl::memory(deconv_pd.src_desc(), engine);
        dnnl::reorder(user_src_mem, deconv_src_mem)
                .execute(stream, user_src_mem, deconv_src_mem);
    }

    if (deconv_pd.weights_desc() != user_weights_mem.get_desc()) {
        deconv_weights_mem = dnnl::memory(deconv_pd.weights_desc(), engine);
        dnnl::reorder(user_weights_mem, deconv_weights_mem)
                .execute(stream, user_weights_mem, deconv_weights_mem);
    }

    if (deconv_pd.dst_desc() != user_dst_mem.get_desc()) {
        deconv_dst_mem = dnnl::memory(deconv_pd.dst_desc(), engine);
    }

    auto conv_prim = dnnl::deconvolution_forward(deconv_pd);

    std::unordered_map<int, dnnl::memory> conv_args;
    conv_args.insert({DNNL_ARG_SRC, deconv_src_mem});
    conv_args.insert({DNNL_ARG_WEIGHTS, deconv_weights_mem});
    conv_args.insert({DNNL_ARG_BIAS, user_bias_mem});
    conv_args.insert({DNNL_ARG_DST, deconv_dst_mem});

    conv_prim.execute(stream, conv_args);

    if (deconv_pd.dst_desc() != user_dst_mem.get_desc()) {
        dnnl::reorder(deconv_dst_mem, user_dst_mem)
                .execute(stream, deconv_dst_mem, user_dst_mem);
    } else
        user_dst_mem = deconv_dst_mem;
    
    stream.wait();
    
    return dst_data;
}

    
}//ops

} //mt 


