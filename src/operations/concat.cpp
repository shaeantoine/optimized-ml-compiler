#include "operations/concat.hpp"
#include "operator_registry.hpp"
#include "ir.hpp"
#include <iostream>


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

    // Debug statements
    std::cout << "ConcatOperator::compute()\n";
    std::cout << " - Axis: " << axis << "\n";
    std::cout << " - Number of inputs: " << input_tensors.size() << "\n";

    for (size_t i = 0; i < input_tensors.size(); ++i) {
        std::cout << "   Input tensor " << i << " shape: [";
        for (size_t d = 0; d < input_tensors[i].shape.size(); ++d) {
            std::cout << input_tensors[i].shape[d];
            if (d < input_tensors[i].shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]\n";
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

    std::vector<float> output_data(output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3]); 
    size_t output_offset = 0;

    int64_t concat_axis_offset = 0;
    for (const auto& tensor : input_tensors) {
        const auto& in_shape = tensor.shape;
        size_t outer = 1, inner = 1;
        for (size_t i = 0; i < axis; ++i) outer *= in_shape[i];
        for (size_t i = axis + 1; i < rank; ++i) inner *= in_shape[i];
        int64_t copy_dim = in_shape[axis];
        size_t block_size = copy_dim * inner;

        for (size_t o = 0; o < outer; ++o) {
            size_t input_index = o * block_size;
            size_t output_index = o * (output_shape[axis] * inner) + concat_axis_offset * inner;

            std::copy(
                tensor.data.begin() + input_index,
                tensor.data.begin() + input_index + block_size,
                output_data.begin() + output_index
            );
        }

        concat_axis_offset += copy_dim;
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