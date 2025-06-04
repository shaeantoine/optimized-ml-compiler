#pragma once

#include "../operator.hpp"

class ConvOperator : public IROperator {
public: 
    void compute(const IRNode& node, ExecutionContext& context) override;
};