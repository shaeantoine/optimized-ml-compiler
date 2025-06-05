#include "operations/softmax.hpp"
#include "operator_registry.hpp"
#include <iostream>
#include <cmath>

void SoftmaxOperator::compute(const IRNode& node, ExecutionContext& context) {
    const auto& input = context.get_tensor(node.inputs[0]);
    const auto& in_shape = input.shape;

    // Debug statements
    std::cout << "SoftmaxOperator::compute()\n";
    std::cout << " - Input shape: ";
    for (auto d : input.shape) std::cout << d << " ";
    std::cout << "\n";
    

    // Accept 4D inputs shaped like [N, C, 1, 1] — flatten them to [N, C]
    if (in_shape.size() == 4 && in_shape[2] == 1 && in_shape[3] == 1) {
        int N = in_shape[0];
        int C = in_shape[1];
        std::vector<float> output_data(N * C);
        for (int n = 0; n < N; ++n) {
            float max_val = -std::numeric_limits<float>::infinity();
            for (int c = 0; c < C; ++c) {
                int idx = ((n * C + c) * 1 + 0) * 1 + 0;
                max_val = std::max(max_val, input.data[idx]);
            }

            float sum_exp = 0.0f;
            for (int c = 0; c < C; ++c) {
                int idx = ((n * C + c) * 1 + 0) * 1 + 0;
                sum_exp += std::exp(input.data[idx] - max_val);
            }

            for (int c = 0; c < C; ++c) {
                int idx = ((n * C + c) * 1 + 0) * 1 + 0;
                output_data[n * C + c] = std::exp(input.data[idx] - max_val) / sum_exp;
            }
        }

        context.set_tensor(node.outputs[0], Tensor(output_data, {N, C}));
        return;
    }

    
    if (in_shape.size() != 2) {
        throw std::runtime_error("Softmax: Only 2D tensors or [N, C, 1, 1] supported");
    }

    int N = in_shape[0];
    int C = in_shape[1];
    std::vector<float> output_data(N * C);

    for (int n = 0; n < N; ++n) {
        float max_val = -std::numeric_limits<float>::infinity();
        for (int c = 0; c < C; ++c) {
            max_val = std::max(max_val, input.data[n * C + c]);
        }

        float sum_exp = 0.0f;
        for (int c = 0; c < C; ++c) {
            sum_exp += std::exp(input.data[n * C + c] - max_val);
        }

        for (int c = 0; c < C; ++c) {
            output_data[n * C + c] = std::exp(input.data[n * C + c] - max_val) / sum_exp;
        }
    }

    context.set_tensor(node.outputs[0], Tensor(output_data, {N, C}));
}

static bool registered = [] {
    OperatorRegistry::instance().register_operator("Softmax", [] {
        return std::make_unique<SoftmaxOperator>();
    });
    return true;
}();
