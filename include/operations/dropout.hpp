#pragma once

#include "../operator.hpp"

class DropoutOperator : public IROperator {
public: 
    void compute(const IRNode& node, ExecutionContext& context) override;
};