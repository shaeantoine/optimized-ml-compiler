#ifndef IR_HPP 
#define IR_HPP

#include <string>
#include <iostream>
#include <vector>
#include <unordered_map>

struct IRNode {
    std::string name;
    std::string op_type;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;

    IRNode(const std::string& name,
           const std::string& op_type,
           const std::vector<std::string>& inputs,
           const std::vector<std::string>& outputs)
        : name(name), op_type(op_type), inputs(inputs), outputs(outputs) {}
};

struct IRGraph {
    std::unordered_map<std::string, IRNode> nodes;

    void add_node(const IRNode& node);
};

#endif