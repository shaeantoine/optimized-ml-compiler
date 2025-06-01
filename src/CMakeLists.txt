add_executable(opty
    main.cpp
    ir.cpp
    tensor.cpp
    execution_context.cpp
    operator_registry.cpp
)

target_include_directories(opty PRIVATE
    ${Protobuf_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/third_party/onnx
    ${CMAKE_SOURCE_DIR}/include
)

find_package(Protobuf REQUIRED)
target_link_libraries(opty
    onnx_proto
    ${Protobuf_LIBRARIES}
)

add_subdirectory(operations)