#include "operator_registry.hpp"
#include <stdexcept>

OperatorRegistry& OperatorRegistry::instance() {
    static OperatorRegistry instance;
    return instance;
}

void OperatorRegistry::register_operator(const std::string& op_type, Creator creator) {
    registry[op_type] = creator;
}

std::unique_ptr<IROperator> OperatorRegistry::create_operator(const std::string& op_type) {
    auto it = registry.find(op_type);
    if (it == registry.end()) {
        throw std::runtime_error("Unknown operator: " + op_type);
    }
    return it->second();
}