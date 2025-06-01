#pragma once

#include "../operator.hpp"

class SigmoidOperator : public IROperator {
public: 
    void compute(const IRNode& node, ExecutionContext& context) override;
};