#include "ir.hpp"
#include "onnx_to_ir.hpp"
#include "context.hpp"
#include "execution_context.hpp"
#include "tensor.hpp"
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

    Tensor t({1.0, 2.0, 3.0, 4.0}, {2, 2});
    std::cout << "Tensor size: " << t.size() << std::endl;
    std::cout << "First element: " << t[0] << std::endl;

    ExecutionContext ctx;
    ctx.set_tensor("input1", Tensor({10.0, 20.0}, {2}));

    if (ctx.has_tensor("input1")) {
        std::cout << "input1[0] = " << ctx.get_tensor("input1")[0] << std::endl;
    }

    return 0;
}