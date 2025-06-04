#pragma once

#include "../operator.hpp"

class SoftmaxOperator : public IROperator {
public: 
    void compute(const IRNode& node, ExecutionContext& context) override;
};