add_library(MTensor SHARED)

target_sources(MTensor PRIVATE

  #tensorImp 
  ${CMAKE_CURRENT_SOURCE_DIR}/core/tensorImpl.cpp



  ##############################################
  #                   UTILS                    #
  ##############################################

  ${CMAKE_CURRENT_SOURCE_DIR}/ops/utils/tensor_print.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/utils/broadcast.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/utils/tensor_iterator.cpp



  ##############################################
  #           TENSOR      OPERATIONS           #
  ##############################################

  ${CMAKE_CURRENT_SOURCE_DIR}/ops/ops.cpp

  #view_ops
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/view/view.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/view/transpose.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/view/permute.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/view/squeeze.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/view/unsqueeze.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/view/narrow.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/view/slice.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/view/expand.cpp

  #memory_ops 
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/memory/clone.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/memory/contiguous.cpp

  #join operations
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/joining/cat.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/joining/stack.cpp

  #eltwise_ops
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/eltwise/exp.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/eltwise/tanh.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/eltwise/linear.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/eltwise/abs.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/eltwise/log.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/eltwise/clip.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/eltwise/pow.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/eltwise/relu.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/eltwise/sigmoid.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/eltwise/sqrt.cpp

  #binary_ops
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/binary/add.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/binary/sub.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/binary/mul.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/binary/div.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/binary/eq.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/binary/ne.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/binary/ge.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/binary/gt.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/binary/le.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/binary/lt.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/binary/max.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/binary/min.cpp

  #reduction_ops
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/reduction/mean.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/reduction/max_reduction.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/reduction/min_reduction.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/reduction/sum.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/reduction/mul_reduction.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/reduction/norm_lp_sum.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/reduction/norm_lp_power_p_sum.cpp

  #linear 
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/linear/matmul.cpp

  #softmax 
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/softmax/softmax.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/softmax/log_softmax.cpp

  #pooling 
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/pooling/pooling_max_1d.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/pooling/pooling_max_2d.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/pooling/pooling_max_3d.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/pooling/pooling_avg_1d.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/pooling/pooling_avg_2d.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/pooling/pooling_avg_3d.cpp

  #conv
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/convolution/conv_1d.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/convolution/conv_2d.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/convolution/conv_3d.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/convolution/deconv_1d.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/convolution/deconv_2d.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/convolution/deconv_3d.cpp

  #normalization
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/normalization/batch_normalization.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/normalization/layer_normalization.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ops/cpu/normalization/group_normalization.cpp

  
  #################################################
  #                     GRAPHS                    #
  #################################################


  #grad_graph 
  ${CMAKE_CURRENT_SOURCE_DIR}/graph/grad_graph.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/graph/export.cpp


) 


set_target_properties(MTensor PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

target_compile_definitions(MTensor PRIVATE MTENSOR_EXPORTS) 

target_include_directories(MTensor PUBLIC ${CMAKE_SOURCE_DIR}/include)

target_link_libraries(MTensor PUBLIC  dnnl) #public cause of the ops.hpp(which will be used in the target that will link to Mtensor) needs dnnl.hpp.

target_link_libraries(MTensor PRIVATE OpenMP::OpenMP_CXX)




