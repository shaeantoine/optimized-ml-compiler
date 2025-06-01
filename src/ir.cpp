#include "ir.hpp"

IRNode::IRNode() = default; 

IRNode::IRNode(const std::string& name,
               const std::string& op_type,
               const std::vector<std::string>& inputs,
               const std::vector<std::string>& outputs):
            name(name), op_type(op_type), inputs(inputs), outputs(outputs) {}

void IRGraph::add_node(const IRNode& node) {
    nodes[node.name] = node;
}