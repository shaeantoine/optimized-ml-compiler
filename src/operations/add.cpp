#include "operations/add.hpp"
#include "operator_registry.hpp"

void AddOperator::compute(const IRNode& node, ExecutionContext& context) {
    const auto& A = context.get_tensor(node.inputs[0]);
    const auto& B = context.get_tensor(node.inputs[1]);

    if (A.data.size() != B.data.size()) {
        throw std::runtime_error("Add: mismatched tensor input sizes");
    }

    std::vector<float> result_data;
    for (size_t i = 0; i < A.data.size(); ++i) {
        result_data.push_back(A.data[i] + B.data[i]);
    }

    Tensor out(result_data, A.shape);
    context.set_tensor(node.outputs[0], out);
}

static bool registered = [] {
    OperatorRegistry::instance().register_operator("Add", [] {
        return std::make_unique<AddOperator>();
    });
    return true;
}();