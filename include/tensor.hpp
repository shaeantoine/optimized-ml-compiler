#pragma once

#include <cstdint>
#include <vector>

struct Tensor {
    std::vector<float> data;
    std::vector<int64_t> shape; 

    Tensor() = default;
    Tensor(
        const std::vector<float>& data,
        const std::vector<int64_t>& shape
    );

    size_t size() const;
    float& operator[](size_t index);
    const float& operator[](size_t index) const;
};