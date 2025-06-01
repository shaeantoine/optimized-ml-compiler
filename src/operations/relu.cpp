#include "operations/relu.hpp"
#include "operator_registry.hpp"
#include <algorithm>

void ReluOperator::compute(const IRNode& node, ExecutionContext& context) {
    const auto& input = context.get_tensor(node.inputs[0]);
    std::vector<float> result;

    for (float x : input.data) {
        result.push_back(std::max(0.0f, x));
    }

    Tensor out(result, input.shape);
    context.set_tensor(node.outputs[0], out);
}

static bool registered = [] {
    OperatorRegistry::instance().register_operator("Relu", [] {
        return std::make_unique<ReluOperator>();
    });
    return true;
}();