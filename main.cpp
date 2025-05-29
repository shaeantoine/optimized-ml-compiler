#include "ir.hpp"
#include "onnx_to_ir.hpp"
#include <onnx/onnx_pb.h>
#include <fstream>
#include <iostream>

int main() {
    onnx::ModelProto model;
    std::ifstream input("../model/gptneox_Opset16.onnx", std::ios::binary);
    if (!model.ParseFromIstream(&input)) {
        std::cerr << "Failed to load ONNX model.\n";
        return 1;
    }

    IRGraph ir = parse_onnx_model(model);

    std::cout << "Parsed IR nodes:\n";
    for (const auto& pair : ir.nodes) {
        const IRNode& node = pair.second;
        std::cout << "- " << node.name << " (" << node.op_type << ")\n";
    }

    return 0;
}