//
// Created by Orlando on 8/23/24.
//
#include "tensor_utils.h"

float** allocate2DArray(TensorDimensions& tensor_dims, void* buffer) {
    int rows = tensor_dims.dims[0];
    int cols = tensor_dims.dims[1];

    // Allocate memory for row pointers
    float** array2D = new float*[rows];

    // Set row pointers to appropriate places in the buffer
    for (int i = 0; i < rows; ++i) {
        array2D[i] = static_cast<float*>(buffer) + i * cols;
    }

    return array2D;
}