#include "execution_context.hpp"
#include <stdexcept>

void ExecutionContext::set_tensor(const std::string& name, const Tensor& tensor) {
    tensors[name] = tensor;
}

Tensor& ExecutionContext::get_tensor(const std::string& name) {
    if (!has_tensor(name)) {
        throw std::runtime_error("Tensor not found: " + name);
    }
    return tensors[name];
}

const Tensor& ExecutionContext::get_tensor(const std::string& name) const {
    if (!has_tensor(name)) {
        throw std::runtime_error("Tensor not found: " + name);
    }
    return tensors.at(name);
}

bool ExecutionContext::has_tensor(const std::string& name) const {
    return tensors.find(name) != tensors.end();
}
