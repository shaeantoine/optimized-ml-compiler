#include "../include/execution_context.hpp"
#include "../include/operator_registry.hpp"
#include "../include/onnx_to_ir.hpp"
#include "../include/operator.hpp"
#include "../include/tensor.hpp"
#include "../include/ir.hpp"

#include "../include/operations/add.hpp"
#include "../include/operations/relu.hpp"
#include "../include/operations/matmul.hpp"
#include "../include/operations/maxpool.hpp"
#include "../include/operations/softmax.hpp"
#include "../include/operations/sigmoid.hpp"

#include <onnx/onnx_pb.h>
#include <fstream>
#include <iostream>

// Testing Adding two tensors
void evaluate_graph(const IRGraph& graph, ExecutionContext& context) {
    for (const auto& pair : graph.nodes) {
        const auto& node = pair.second;
        auto op = OperatorRegistry::instance().create_operator(node.op_type);
        op->compute(node, context);
    }
}

int main() {
    onnx::ModelProto model;
    std::ifstream input("../model/squeezenet1.0-3.onnx", std::ios::binary);
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

    // Testing registering tensors
    // Tensor t({1.0, 2.0, 3.0, 4.0}, {2, 2});
    // std::cout << "Tensor size: " << t.size() << std::endl;
    // std::cout << "First element: " << t[0] << std::endl;

    // Testing execution context
    // ExecutionContext ctx;
    // ctx.set_tensor("input1", Tensor({10.0, 20.0}, {2}));

    // if (ctx.has_tensor("input1")) {
    //     std::cout << "input1[0] = " << ctx.get_tensor("input1")[0] << std::endl;
    // }

    // Testing simple evaluation task
    evaluate_graph(ir, ctx);

    return 0;
}