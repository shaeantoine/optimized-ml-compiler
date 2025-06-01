cmake_minimum_required(VERSION 3.15)
project(opty)

set(CMAKE_CXX_STANDARD 17)

# Fetch Protobuf (optional if already installed)
find_package(Protobuf REQUIRED)

# Add ONNX subdirectory (assumes ONNX is a git submodule)
add_subdirectory(third_party/onnx
                 src/operations)

include_directories(
    ${Protobuf_INCLUDE_DIRS}
    third_party/onnx
)

target_sources(opty PRIVATE
               operator_registry.cpp
               operations/add.cpp
               operations/matmul.cpp
               operations/sigmoid.cpp
               operations/relu.cpp)

add_executable(opty 
               main.cpp
               ir.cpp
               tensor.cpp
               execution_context.cpp)



# Link against ONNX and Protobuf
target_link_libraries(opty
    onnx_proto
    ${Protobuf_LIBRARIES}
)
