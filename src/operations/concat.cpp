#include "operations/concat.hpp"
#include "operator_registry.hpp"
#include "ir.hpp"

void ConcatOperator::compute(const IRNode& node, ExecutionContext& context) {
    int axis = 0;
    if (node.attributes.count("axis")) {
        axis = node.attributes.at("axis").i();
    }

    if (node.inputs.empty()) {
        throw std::runtime_error("Concat: no input tensors provided.");
    }

    std::vector<Tensor> input_tensors;
    for (const auto& input_name : node.inputs) {
        if (!context.has_tensor(input_name)) {
            throw std::runtime_error("Concat: input tensor not found: " + input_name);
        }
        input_tensors.push_back(context.get_tensor(input_name));
    }

    const size_t num_inputs = input_tensors.size();
    const size_t rank = input_tensors[0].shape.size();
    if (axis < 0 || static_cast<size_t>(axis) >= rank) {
        throw std::runtime_error("Concat: axis out of bounds");
    }

    std::vector<int64_t> output_shape = input_tensors[0].shape;
    int concat_dim = output_shape[axis];

    for (size_t i = 1; i < num_inputs; ++i) {
        const auto& shape = input_tensors[i].shape;
        for (size_t d = 0; d < rank; ++d) {
            if (d == static_cast<size_t>(axis)) continue;
            if (shape[d] != output_shape[d]) {
                throw std::runtime_error("Concat: tensor shapes don't match on non-concat axes");
            }
        }
        concat_dim += shape[axis];
    }

    output_shape[axis] = concat_dim;

    std::vector<float> output_data;
    std::vector<int> strides(rank, 1);
    for (int i = rank - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * output_shape[i + 1];
    }

    size_t offset = 0;
    for (const auto& tensor : input_tensors) {
        output_data.insert(output_data.end(), tensor.data.begin(), tensor.data.end());
        offset += tensor.data.size();
    }

    Tensor output_tensor(output_data, output_shape);
    context.set_tensor(node.outputs[0], output_tensor);
    
}

static bool registered = [] {
    OperatorRegistry::instance().register_operator("Concat", [] {
        return std::make_unique<ConcatOperator>();
    });
    return true;
}();