#include "tensor.hpp"

Tensor::Tensor() = default;

Tensor::Tensor(
    const std::vector<std::float>& data,
    const std::vector<std::int64_t>& shape):
    data(data), shape(shape) {}