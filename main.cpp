#include <onnx/onnx_pb.h>
#include <fstream>
#include <iostream>

int main() {
    onnx::ModelProto model;
    std::ifstream input("gptneox_Opset16.onnx", std::ios::binary);
    if (!model.ParseFromIstream(&input)) {
        std::cerr << "Failed to load ONNX model.\n";
        return 1;
    }
    std::cout << "Model loaded! IR version: " << model.ir_version() << "\n";

    const onnx::GraphProto& graph = model.graph();

    for (const auto& node : graph.node()) {
        std::cout << "Op: " << node.op_type() << "\n";
        for (const auto& input : node.input())
            std::cout << "  Input: " << input << "\n";
        for (const auto& output : node.output())
            std::cout << "  Output: " << output << "\n";
    }    

    return 0;
}