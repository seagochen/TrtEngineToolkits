//
// Created by vipuser on 8/23/24.
//

#ifndef JETSON_INFER_CU_MEMORY_CUH
#define JETSON_INFER_CU_MEMORY_CUH

#include <cuda_runtime.h>

// 切割数组
__global__ void splitArrayKernel(const int *input, int *output, int arraySize, int chunkSize);

// 合并数组
__global__ void mergeArrayKernel(const int *input, int *output, int arraySize, int chunkSize);

// 提取数组片段
__global__ void extractSegmentKernel(const int *input, int *output, int startIndex, int segmentLength);

#endif //JETSON_INFER_CU_MEMORY_CUH
