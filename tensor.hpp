#include <iostream>
#include <vector>

struct Tensor {
    std::vector<std::float> data;
    std::vector<int64_t> shape; 

    Tensor();
    Tensor(
        const std::vector<std::float>& data,
        const std::vector<std::int64_t>& shape
    )
}