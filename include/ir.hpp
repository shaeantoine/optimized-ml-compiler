#ifndef IR_HPP 
#define IR_HPP

#include <map>
#include <string>
#include <vector>
#include <unordered_map>
#include <onnx/onnx_pb.h>

struct IRNode {
    std::string name;
    std::string op_type;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::map<std::string, onnx::AttributeProto> attributes;

    IRNode();
    IRNode(const std::string& name,
           const std::string& op_type,
           const std::vector<std::string>& inputs,
           const std::vector<std::string>& outputs);
};

struct IRGraph {
    std::unordered_map<std::string, IRNode> nodes;

    void add_node(const IRNode& node);
};

#endif