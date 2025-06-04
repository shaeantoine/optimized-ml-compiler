#include "operations/conv.hpp"
#include "operator_registry.hpp"

void ConvOperator::compute(const IRNode& node, ExecutionContext& context) {
    const auto& input = context.get_tensor(node.inputs[0]);      // [N, C_in, H_in, W_in]
    const auto& weights = context.get_tensor(node.inputs[1]);    // [C_out, C_in, kH, kW]

    const auto& in_shape = input.shape;
    const auto& w_shape = weights.shape;

    if (in_shape.size() != 4 || w_shape.size() != 4) {
        throw std::runtime_error("Conv: expected 4D input and weights");
    }

    int N = in_shape[0], C_in = in_shape[1], H = in_shape[2], W = in_shape[3];
    int C_out = w_shape[0], kC = w_shape[1], kH = w_shape[2], kW = w_shape[3];

    if (C_in != kC) {
        throw std::runtime_error("Conv: input channels do not match");
    }

    // Parse attributes
    int stride_h = 1, stride_w = 1;
    if (node.attributes.count("strides")) {
        stride_h = node.attributes.at("strides").ints(0);
        stride_w = node.attributes.at("strides").ints(1);
    }

    int pad_h = 0, pad_w = 0;
    if (node.attributes.count("pads")) {
        pad_h = node.attributes.at("pads").ints(0); // pads = [pad_top, pad_left, pad_bottom, pad_right]
        pad_w = node.attributes.at("pads").ints(1);
    }

    // Compute output dimensions
    int out_H = (H + 2 * pad_h - kH) / stride_h + 1;
    int out_W = (W + 2 * pad_w - kW) / stride_w + 1;
    std::vector<float> output_data(N * C_out * out_H * out_W, 0.0f);

    for (int n = 0; n < N; ++n) {
        for (int co = 0; co < C_out; ++co) {
            for (int ho = 0; ho < out_H; ++ho) {
                for (int wo = 0; wo < out_W; ++wo) {
                    float sum = 0.0f;
                    for (int ci = 0; ci < C_in; ++ci) {
                        for (int kh = 0; kh < kH; ++kh) {
                            for (int kw = 0; kw < kW; ++kw) {
                                int h_in = ho * stride_h - pad_h + kh;
                                int w_in = wo * stride_w - pad_w + kw;
                                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                                    int in_idx = ((n * C_in + ci) * H + h_in) * W + w_in;
                                    int w_idx = ((co * C_in + ci) * kH + kh) * kW + kw;
                                    sum += input.data[in_idx] * weights.data[w_idx];
                                }
                            }
                        }
                    }
                    int out_idx = ((n * C_out + co) * out_H + ho) * out_W + wo;
                    output_data[out_idx] = sum;
                }
            }
        }
    }

    if (node.inputs.size() == 3) {
        const auto& bias = context.get_tensor(node.inputs[2]); // [C_out]
        for (int n = 0; n < N; ++n) {
            for (int co = 0; co < C_out; ++co) {
                for (int ho = 0; ho < out_H; ++ho) {
                    for (int wo = 0; wo < out_W; ++wo) {
                        int out_idx = ((n * C_out + co) * out_H + ho) * out_W + wo;
                        output_data[out_idx] += bias.data[co];
                    }
                }
            }
        }
    }    

    context.set_tensor(node.outputs[0], Tensor(output_data, {N, C_out, out_H, out_W}));
}

static bool registered = [] {
    OperatorRegistry::instance().register_operator("Conv", [] {
        return std::make_unique<ConvOperator>();
    });
    return true;
}();