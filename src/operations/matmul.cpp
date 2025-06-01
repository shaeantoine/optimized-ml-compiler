#include "operations/matmul.hpp"
#include "operator_registry.hpp"
#include <stdexcept>

void MatmulOperator::compute(const IRNode& node, ExecutionContext& context) {
    const auto& A = context.get_tensor(node.inputs[0]);
    const auto& B = context.get_tensor(node.inputs[1]);

    if (A.shape.size() != 2 || B.shape.size() != 2) {
        throw std::runtime_error("Matmul: Only 2D tensors are supported");
    }

    int m = A.shape[0], k = A.shape[1];
    int k2 = B.shape[0], n = B.shape[1];

    if (k != k2) {
        throw std::runtime_error("Matmul: Incompatible shapes");
    }

    std::vector<float> result(m * n, 0.0f);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int kk = 0; kk < k; ++kk) {
                result[i * n + j] += A.data[i * k + kk] * B.data[kk * n + j];
            }
        }
    }

    Tensor out(result, {m, n});
    context.set_tensor(node.outputs[0], out);
}

static bool registered = [] {
    OperatorRegistry::instance().register_operator("Matmul", [] {
        return std::make_unique<MatmulOperator>();
    });
    return true;
}();
