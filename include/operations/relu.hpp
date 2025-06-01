#pragma once

#include "../operator.hpp"

class ReluOperator : public IROperator {
public: 
    void compute(const IRNode& node, ExecutionContext& context) override;
};