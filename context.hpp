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