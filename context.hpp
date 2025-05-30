#pragma once
#include <unordered_map>
#include <string>
#include "tensor.hpp"

struct ExecutionContext {
    std::unordered_map<std::string, Tensor> tensors;

    bool has(const std::string& name) const; 
    Tensor& get(const std::string& name);
    void set(const std::string& name, const Tensor& tensor);
};

// void evaluate_add(const IRNode& node, ExecutionContext& ctx) {
//     const Tensor& a = ctx.get(node.inputs[0]);
//     const Tensor& b = ctx.get(node.inputs[1]);
//     Tensor result;
//     result.shape = a.shape;
//     result.data.resize(a.size());

//     for (size_t i = 0; i < a.size(); ++i) {
//         result.data[i] = a.data[i] + b.data[i];
//     }

//     ctx.set(node.outputs[0], result);
// }
