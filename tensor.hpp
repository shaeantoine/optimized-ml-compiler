#include <iostream>
#include <vector>

struct Tensor {
    std::vector<std::float> data;
    std::vector<int64_t> shape; 

    Tensor() = default;
    Tensor(
        const std::vector<std::float>& data,
        const std::vector<std::int64_t>& shape
    );

    size_t size() const;
    float& operator[](size_t index);
    const float& operator[](size_t index) const;
}