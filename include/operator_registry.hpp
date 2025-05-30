#pragma once

#include "operator.hpp"
#include <functional>
#include <unordered_map>
#include <memory>

class OperatorRegistry {
public: 
    using Creator = std::function<std::unique_ptr<IROperator>()>;

    static OperatorRegistry& instance();

    void register_operator(const std::string& op_type, Creator creator);

    std::unique_ptr<IROperator> create_operator(const std::string& op_type);

private:
    std::unordered_map<std::string, Creator> registry;
};