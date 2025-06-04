#pragma once

#include "../operator.hpp"

class ConcatOperator : public IROperator {
public: 
    void compute(const IRNode& node, ExecutionContext& context) override;
};