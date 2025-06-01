#include "operations/sigmoid.hpp"
#include "operator_registry.hpp"
#include <cmath>

void SigmoidOperator::compute(const IRNode& node, ExecutionContext& context) {
    const auto& input = context.get_tensor(node.inputs[0]);
    std::vector<float> result;

    for (float x : input.data) {
        result.push_back(1.0f / (1.0f + std::exp(-x)));
    }

    Tensor out(result, input.shape);
    context.set_tensor(node.outputs[0], out);
}

static bool registered = [] {
    OperatorRegistry::instance().register_operator("Sigmoid", [] {
        return std::make_unique<SigmoidOperator>();
    });
    return true;
}();
