#include "tensor.hpp"

//Tensor::Tensor() = default;

Tensor::Tensor(
    const std::vector<float>& data,
    const std::vector<int64_t>& shape):
    data(data), shape(shape) {}

size_t Tensor::size() const {
    return data.size();
}

float& Tensor::operator[](size_t index) {
    return data[index];
}

const float& Tensor::operator[](size_t index) const {
    return data[index];
}