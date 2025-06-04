#pragma once

#include "../operator.hpp"

class GlobalaveragepoolOperator : public IROperator {
public: 
    void compute(const IRNode& node, ExecutionContext& context) override;
};