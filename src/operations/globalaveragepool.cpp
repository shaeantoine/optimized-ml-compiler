#include "operations/globalaveragepool.hpp"
#include "operator_registry.hpp"
#include <stdexcept>

void GlobalaveragepoolOperator::compute(const IRNode& node, ExecutionContext& context) {
    const auto& input = context.get_tensor(node.inputs[0]);
    const auto& shape = input.shape;

    if (shape.size() != 4) {
        throw std::runtime_error("GlobalAveragePool expects 4D input [N, C, H, W]");
    }

    int N = shape[0];
    int C = shape[1];
    int H = shape[2];
    int W = shape[3];

    std::vector<float> result_data;
    result_data.reserve(N * C);

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            float sum = 0.0f;
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    int index = ((n * C + c) * H + h) * W + w;
                    sum += input.data[index];
                }
            }
            float avg = sum / (H * W);
            result_data.push_back(avg);
        }
    }

    Tensor output(result_data, {N, C, 1, 1});
    context.set_tensor(node.outputs[0], output);
}

static bool registered = [] {
    OperatorRegistry::instance().register_operator("GlobalAveragePool", [] {
        return std::make_unique<GlobalaveragepoolOperator>();
    });
    return true;
}();

