#pragma once

#include "../operator.hpp"

class MaxpoolOperator : public IROperator {
public: 
    void compute(const IRNode& node, ExecutionContext& context) override;
};