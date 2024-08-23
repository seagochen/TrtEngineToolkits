//
// Created by Orlando on 8/23/24.
//

#ifndef JETSON_INFER_TENSOR_UTILS_H
#define JETSON_INFER_TENSOR_UTILS_H

#include <iostream>
#include <cuda_runtime.h>
#include <vector>

// Define a struct to store tensor dimensions
struct TensorDimensions {
    std::vector<int> dims;
    size_t size;
};

float** allocate2DArray(TensorDimensions& tensor_dims, void* buffer);

#endif //JETSON_INFER_TENSOR_UTILS_H
