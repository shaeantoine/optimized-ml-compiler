#pragma once

#include "../operator.hpp"

class AddOperator : public IROperator {
public: 
    void compute(const IRNode& node, ExecutionContext& context) override;
};