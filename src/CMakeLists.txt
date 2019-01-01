# MNIST Example
add_executable(mnist
                mnist.cpp)

target_compile_features(mnist PUBLIC cxx_range_for)

target_link_libraries(mnist ${TORCH_LIBRARIES})
