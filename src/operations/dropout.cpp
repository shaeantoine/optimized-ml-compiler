#include "operations/dropout.hpp"
#include "operator_registry.hpp"
#include "ir.hpp"

void DropoutOperator::compute(const IRNode& node, ExecutionContext& context) {
    const std::string& input_name = node.inputs[0];
    const std::string& output_name = node.outputs[0];

    if (!context.has_tensor(input_name)) {
        throw std::runtime_error("Input tensor not found for Dropout: " + input_name);
    }

    const Tensor& input = context.get_tensor(input_name);
    Tensor output = input;
    context.set_tensor(output_name, output);
}

static bool registered = [] {
    OperatorRegistry::instance().register_operator("Dropout", [] {
        return std::make_unique<DropoutOperator>();
    });
    return true;
}();