#include "../include/execution_context.hpp"
#include "../include/operator_registry.hpp"
#include "../include/onnx_to_ir.hpp"
#include "../include/operator.hpp"
#include "../include/tensor.hpp"
#include "../include/ir.hpp"

#include "../include/operations/add.hpp"
#include "../include/operations/relu.hpp"
#include "../include/operations/conv.hpp"
#include "../include/operations/matmul.hpp"
#include "../include/operations/concat.hpp"
#include "../include/operations/maxpool.hpp"
#include "../include/operations/softmax.hpp"
#include "../include/operations/sigmoid.hpp"
#include "../include/operations/globalaveragepool.hpp"

#include <onnx/onnx_pb.h>
#include <fstream>
#include <iostream>

// Running the ONNX model graph 
void evaluate_graph(const IRGraph& graph, ExecutionContext& context) {
    for (const auto& pair : graph.nodes) {
        const auto& node = pair.second;
        auto op = OperatorRegistry::instance().create_operator(node.op_type);
        op->compute(node, context);
    }
}

int main() {

    // === Load input ONNX file ===
    onnx::ModelProto model;
    std::ifstream input("../model/squeezenet1.0-3.onnx", std::ios::binary);
    if (!model.ParseFromIstream(&input)) {
        std::cerr << "Failed to load ONNX model.\n";
        return 1;
    }

    // === Create input graph ===
    IRGraph ir = parse_onnx_model(model);

    std::cout << "Parsed IR nodes:\n";
    for (const auto& pair : ir.nodes) {
        const IRNode& node = pair.second;
        std::cout << "- " << node.name << " (" << node.op_type << ")\n";
    }

    // === Create and populate input tensors ===
    ExecutionContext ctx;

    const auto& onnx_graph = model.graph();
    for (const auto& input_tensor : onnx_graph.input()) {
        const std::string& name = input_tensor.name();
        const auto& tensor_type = input_tensor.type().tensor_type();
        const auto& shape_proto = tensor_type.shape();

        std::vector<int64_t> shape;
        int size = 1;
        for (const auto& dim : shape_proto.dim()) {
            int dim_size = dim.has_dim_value() ? dim.dim_value() : 1;
            shape.push_back(dim_size);
            size *= dim_size;
        }

        // Fill input with dummy data (e.g., all 1.0f)
        std::vector<float> data(size, 1.0f);
        Tensor context_tensor(data, shape);
        ctx.set_tensor(name, context_tensor);

        std::cout << "Set input tensor: " << name << " [";
        for (size_t i = 0; i < shape.size(); ++i) {
            std::cout << shape[i];
            if (i < shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]\n";
    }

    // Running graph and tensors 
    evaluate_graph(ir, ctx);

    return 0;
}