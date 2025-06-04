#include "operations/softmax.hpp"
#include "operator_registry.hpp"
#include <cmath>

void SoftmaxOperator::compute(const IRNode& node, ExecutionContext& context) {
    const auto& input = context.get_tensor(node.inputs[0]);
    //std::vector<float> result;


    if (input.shape.size() != 2) {
        throw std::runtime_error("Softmax: Only 2D tensors supported (batch_size, num_classes)");
    }

    size_t batch_size = input.shape[0];
    size_t num_classes = input.shape[1];

    std::vector<float> result(input.data.size());

    for (size_t i = 0; i < batch_size; ++i) {
        float max_val = -std::numeric_limits<float>::infinity();

        // Find max for numerical stability
        for (size_t j = 0; j < num_classes; ++j) {
            max_val = std::max(max_val, input.data[i * num_classes + j]);
        }

        float sum = 0.0f;

        // Compute exponentials and sum
        for (size_t j = 0; j < num_classes; ++j) {
            float val = std::exp(input.data[i * num_classes + j] - max_val);
            result[i * num_classes + j] = val;
            sum += val;
        }

        // Normalize
        for (size_t j = 0; j < num_classes; ++j) {
            result[i * num_classes + j] /= sum;
        }
    }
    

    Tensor out(result, input.shape);
    context.set_tensor(node.outputs[0], out);
}

static bool registered = [] {
    OperatorRegistry::instance().register_operator("Softmax", [] {
        return std::make_unique<SoftmaxOperator>();
    });
    return true;
}();
