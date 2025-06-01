cmake_minimum_required(VERSION 3.15)
project(opty)

set(CMAKE_CXX_STANDARD 17)

find_package(Protobuf REQUIRED)

add_subdirectory(third_party/onnx)
add_subdirectory(src)
