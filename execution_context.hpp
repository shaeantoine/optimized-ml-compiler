#pragma once
#include "tensor.hpp"
#include <unordered_map>
#include <string>

class ExecutionContext {
public:
    std::unordered_map<std::string, Tensor> tensors;

    void set_tensor(const std::string& name, const Tensor& tensor);

    Tensor& get_tensor(const std::string& name);

    const Tensor& get_tensor(const std::string& name) const;

    bool has_tensor(const std::string& name) const;
};
