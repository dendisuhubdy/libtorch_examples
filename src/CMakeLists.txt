# MNIST Example
add_executable(mnist
                mnist.cpp)

target_compile_features(mnist PUBLIC cxx_range_for)

target_link_libraries(mnist ${TORCH_LIBRARIES})

set_property(TARGET mnist PROPERTY CXX_STANDARD 11)

add_executable(dcgan dcgan.cpp)

target_link_libraries(dcgan "${TORCH_LIBRARIES}")

set_property(TARGET dcgan PROPERTY CXX_STANDARD 11)
