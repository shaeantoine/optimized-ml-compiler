#ifndef ONNX_TO_IR_HPP
#define ONNX_TO_IR_HPP

#include "ir.hpp"
#include <onnx/onnx_pb.h>
#include <iostream>

IRGraph parse_onnx_model(const onnx::ModelProto& model) {
    IRGraph graph;

    const onnx::GraphProto& onnx_graph = model.graph();

    for (const auto& node : onnx_graph.node()) {
        std::string name = node.name().empty() ? node.output(0) : node.name();
        std::string op_type = node.op_type();

        std::vector<std::string> inputs(node.input().begin(), node.input().end());
        std::vector<std::string> outputs(node.output().begin(), node.output().end());

        IRNode ir_node(name, op_type, inputs, outputs);

        // Gather node attributes
        for (const auto& attr : node.attribute()) {
            ir_node.attributes[attr.name()] = attr;
        }

        graph.add_node(ir_node);
    }

    return graph;
}

#endif