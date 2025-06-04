#include "operations/maxpool.hpp"
#include "operator_registry.hpp"
#include <algorithm>
#include <cmath>

void MaxpoolOperator::compute(const IRNode& node, ExecutionContext& context) {
    const auto& input = context.get_tensor(node.inputs[0]);

    if (input.shape.size() != 4) {
        throw std::runtime_error("MaxPool: Only 4D tensors supported (NCHW)");
    }

    const int N = input.shape[0];
    const int C = input.shape[1];
    const int H = input.shape[2];
    const int W = input.shape[3];

    const int kernel_h = 2;
    const int kernel_w = 2;
    const int stride_h = 2;
    const int stride_w = 2;

    const int out_h = H / kernel_h;
    const int out_w = W / kernel_w;

    std::vector<float> output_data;
    output_data.reserve(N * C * out_h * out_w);

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    float max_val = -std::numeric_limits<float>::infinity();
                    for (int kh = 0; kh < kernel_h; ++kh) {
                        for (int kw = 0; kw < kernel_w; ++kw) {
                            int ih = oh * stride_h + kh;
                            int iw = ow * stride_w + kw;
                            int idx = ((n * C + c) * H + ih) * W + iw;
                            max_val = std::max(max_val, input.data[idx]);
                        }
                    }
                    output_data.push_back(max_val);
                }
            }
        }
    }

    std::vector<int64_t> output_shape = {N, C, out_h, out_w};
    Tensor out(output_data, output_shape);
    context.set_tensor(node.outputs[0], out);
}

static bool registered = [] {
    OperatorRegistry::instance().register_operator("MaxPool", [] {
        return std::make_unique<MaxpoolOperator>();
    });
    return true;
}();