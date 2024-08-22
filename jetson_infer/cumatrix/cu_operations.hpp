#ifndef __CUDA_MATRIX_OP_HPP__
#define __CUDA_MATRIX_OP_HPP__

#include <cuda_runtime.h>


// 切割数组
__global__ void splitArrayKernel(const int *input, int *output, int arraySize, int chunkSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < arraySize) {
        int chunkIndex = idx / chunkSize;
        int localIndex = idx % chunkSize;
        output[chunkIndex * chunkSize + localIndex] = input[idx];
    }
}


// 合并数组
__global__ void mergeArrayKernel(const int *input, int *output, int arraySize, int chunkSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < arraySize) {
        int chunkIndex = idx / chunkSize;
        int localIndex = idx % chunkSize;
        output[idx] = input[chunkIndex * chunkSize + localIndex];
    }
}


// 提取数组片段
 __global__ void extractSegmentKernel(const int *input, int *output, int startIndex, int segmentLength) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < segmentLength) {
        output[idx] = input[startIndex + idx];
    }
}

#endif // __CUDA_MATRIX_OP_HPP__