#pragma once

#include "execution_context.hpp"
#include "ir.hpp"

struct IROperator {
    virtual ~IROperator() = default; 

    virtual void compute(const IRNode& node, ExecutionContext& context) = 0;
};