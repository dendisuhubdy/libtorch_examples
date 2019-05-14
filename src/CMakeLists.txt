# MNIST Example

add_executable(mnist mnist.cpp)

target_compile_features(mnist PUBLIC cxx_range_for)

target_link_libraries(mnist ${TORCH_LIBRARIES})

set_property(TARGET mnist PROPERTY CXX_STANDARD 11)

# DCGAN Example

add_executable(dcgan dcgan.cpp)

target_link_libraries(dcgan "${TORCH_LIBRARIES}")

set_property(TARGET dcgan PROPERTY CXX_STANDARD 11)

# Yolov3 Example

add_executable(yolov3 darknet.cpp yolov3.cpp)

target_link_libraries(yolov3 "${OpenCV_LIBS}")
target_link_libraries(yolov3 "${TORCH_LIBRARIES}")

set_property(TARGET yolov3 PROPERTY CXX_STANDARD 11)
