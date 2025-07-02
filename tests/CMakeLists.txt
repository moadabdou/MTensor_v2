

# add_executable(init_tensorImpl)

# target_sources(init_tensorImpl PRIVATE init_tensorImpl.cpp)

# set_target_properties(init_tensorImpl PROPERTIES 
#     RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
# )

# target_link_libraries( init_tensorImpl PRIVATE  GTest::gtest_main)
# target_link_libraries( init_tensorImpl PRIVATE MTensor )
# include(GoogleTest)
# gtest_discover_tests(init_tensorImpl)


# add_executable(view_ops)

# target_sources(view_ops PRIVATE view_ops.cpp)

# set_target_properties(view_ops PROPERTIES 
#     RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
# )

# target_link_libraries( view_ops PRIVATE  GTest::gtest_main)
# target_link_libraries( view_ops PRIVATE MTensor )
# include(GoogleTest)
# gtest_discover_tests(view_ops)

# add_executable(memory_ops)

# target_sources(memory_ops PRIVATE memory_ops.cpp)

# set_target_properties(memory_ops PROPERTIES 
#     RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
# )

# target_link_libraries( memory_ops PRIVATE  GTest::gtest_main)
# target_link_libraries( memory_ops PRIVATE MTensor )
# include(GoogleTest)
# gtest_discover_tests(memory_ops)

add_executable(dnnl_tests)

target_sources(dnnl_tests PRIVATE dnnl_tests.cpp)

set_target_properties(dnnl_tests PROPERTIES 
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

target_include_directories( dnnl_tests  PRIVATE 
    ${CMAKE_BINARY_DIR}/_deps/dnnl-src/examples
)

target_link_libraries( dnnl_tests PRIVATE  GTest::gtest_main)
target_link_libraries( dnnl_tests PRIVATE MTensor )

include(GoogleTest)
gtest_discover_tests(dnnl_tests)


# add_executable(eltwise_ops)

# target_sources(eltwise_ops PRIVATE eltwise_ops.cpp)

# set_target_properties(eltwise_ops PROPERTIES 
#     RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
# )

# target_link_libraries( eltwise_ops PRIVATE  GTest::gtest_main)
# target_link_libraries( eltwise_ops PRIVATE MTensor )
# include(GoogleTest)
# gtest_discover_tests(eltwise_ops)

# add_executable(broadcast_test)

# target_sources(broadcast_test PRIVATE broadcast_test.cpp)

# set_target_properties(broadcast_test PROPERTIES 
#     RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
# )

# target_link_libraries( broadcast_test PRIVATE  GTest::gtest_main)
# target_link_libraries( broadcast_test PRIVATE MTensor )
# include(GoogleTest)
# gtest_discover_tests(broadcast_test)


# add_executable(binary_ops)

# target_sources(binary_ops PRIVATE binary_ops.cpp)

# set_target_properties(binary_ops PROPERTIES 
#     RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
# )

# target_link_libraries( binary_ops PRIVATE  GTest::gtest_main)
# target_link_libraries( binary_ops PRIVATE MTensor )
# include(GoogleTest)
# gtest_discover_tests(binary_ops)


# add_executable(reduction_ops)

# target_sources(reduction_ops PRIVATE reduction_ops.cpp)

# set_target_properties(reduction_ops PROPERTIES 
#     RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
# )

# target_link_libraries( reduction_ops PRIVATE  GTest::gtest_main)
# target_link_libraries( reduction_ops PRIVATE MTensor )
# include(GoogleTest)
# gtest_discover_tests(reduction_ops)


# add_executable(linear_ops)

# target_sources(linear_ops PRIVATE linear_ops.cpp)

# set_target_properties(linear_ops PROPERTIES 
#     RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
# )

# target_include_directories( linear_ops  PRIVATE 
#     ${CMAKE_BINARY_DIR}/_deps/dnnl-src/examples
# )

# target_link_libraries( linear_ops PRIVATE  GTest::gtest_main)
# target_link_libraries( linear_ops PRIVATE MTensor )

# include(GoogleTest)
# gtest_discover_tests(linear_ops)

# add_executable(softmax_ops)

# target_sources(softmax_ops PRIVATE softmax_ops.cpp)

# set_target_properties(softmax_ops PROPERTIES 
#     RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
# )

# target_include_directories( softmax_ops  PRIVATE 
#     ${CMAKE_BINARY_DIR}/_deps/dnnl-src/examples
# )

# target_link_libraries( softmax_ops PRIVATE  GTest::gtest_main)
# target_link_libraries( softmax_ops PRIVATE MTensor )

# include(GoogleTest)
# gtest_discover_tests(softmax_ops)

# add_executable(pooling)

# target_sources(pooling PRIVATE pooling.cpp)

# set_target_properties(pooling PROPERTIES 
#     RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
# )

# target_include_directories( pooling  PRIVATE 
#     ${CMAKE_BINARY_DIR}/_deps/dnnl-src/examples
# )

# target_link_libraries( pooling PRIVATE  GTest::gtest_main)
# target_link_libraries( pooling PRIVATE MTensor )

# include(GoogleTest)
# gtest_discover_tests(pooling)

# add_executable(conv_tests)

# target_sources(conv_tests PRIVATE conv_tests.cpp)

# set_target_properties(conv_tests PROPERTIES 
#     RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
# )

# target_include_directories( conv_tests  PRIVATE 
#     ${CMAKE_BINARY_DIR}/_deps/dnnl-src/examples
# )

# target_link_libraries( conv_tests PRIVATE  GTest::gtest_main)
# target_link_libraries( conv_tests PRIVATE MTensor )

# include(GoogleTest)
# gtest_discover_tests(conv_tests)

# add_executable(deconv_tests)

# target_sources(deconv_tests PRIVATE deconv_tests.cpp)

# set_target_properties(deconv_tests PROPERTIES 
#     RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
# )

# target_include_directories( deconv_tests  PRIVATE 
#     ${CMAKE_BINARY_DIR}/_deps/dnnl-src/examples
# )

# target_link_libraries( deconv_tests PRIVATE  GTest::gtest_main)
# target_link_libraries( deconv_tests PRIVATE MTensor )

# include(GoogleTest)
# gtest_discover_tests(deconv_tests)


# add_executable(normalization_tests)

# target_sources(normalization_tests PRIVATE normalization_tests.cpp)

# set_target_properties(normalization_tests PROPERTIES 
#     RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
# )

# target_include_directories( normalization_tests  PRIVATE 
#     ${CMAKE_BINARY_DIR}/_deps/dnnl-src/examples
# )

# target_link_libraries( normalization_tests PRIVATE  GTest::gtest_main)
# target_link_libraries( normalization_tests PRIVATE MTensor )

# include(GoogleTest)
# gtest_discover_tests(normalization_tests)

# add_executable(graph_tests)

# target_sources(graph_tests PRIVATE graph_tests.cpp)

# set_target_properties(graph_tests PROPERTIES 
#     RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
# )

# target_include_directories( graph_tests  PRIVATE 
#     ${CMAKE_BINARY_DIR}/_deps/dnnl-src/examples
# )

# target_link_libraries( graph_tests PRIVATE  GTest::gtest_main)
# target_link_libraries( graph_tests PRIVATE MTensor )

# include(GoogleTest)
# gtest_discover_tests(graph_tests)

# add_executable(binary_grad)

# target_sources(binary_grad PRIVATE binary_grad.cpp)

# set_target_properties(binary_grad PROPERTIES 
#     RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
# )

# target_link_libraries( binary_grad PRIVATE  GTest::gtest_main)
# target_link_libraries( binary_grad PRIVATE MTensor )

# include(GoogleTest)
# gtest_discover_tests(binary_grad)


# add_executable(eltwise_grad)

# target_sources(eltwise_grad PRIVATE eltwise_grad.cpp)

# set_target_properties(eltwise_grad PROPERTIES 
#     RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
# )

# target_link_libraries( eltwise_grad PRIVATE  GTest::gtest_main)
# target_link_libraries( eltwise_grad PRIVATE MTensor )
# include(GoogleTest)
# gtest_discover_tests(eltwise_grad)

add_executable(view_grad)

target_sources(view_grad PRIVATE view_grad.cpp)

set_target_properties(view_grad PROPERTIES 
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

target_link_libraries( view_grad PRIVATE  GTest::gtest_main)
target_link_libraries( view_grad PRIVATE MTensor )
include(GoogleTest)
gtest_discover_tests(view_grad)