#pragma once

#include "../operator.hpp"

class MatmulOperator : public IROperator {
public: 
    void compute(const IRNode& node, ExecutionContext& context) override;
};