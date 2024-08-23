//
// Created by vipuser on 8/23/24.
//

#ifndef CU_VECTOR_OPS_H
#define CU_VECTOR_OPS_H

#include <cuda_runtime.h>

// 向量求和
__global__ void sumKernel(const float* d_input, float* d_sum, int size);

// 向量加法
__global__ void vectorAddKernel(const float* d_A, const float* d_B, float* d_C, int size);

// 向量减法
__global__ void vectorSubKernel(const float* d_A, const float* d_B, float* d_C, int size);

// 向量点积
__global__ void vectorDotProductKernel(const float* d_A, const float* d_B, float* d_C, int size);

// 向量叉乘
__global__ void vectorCrossProductKernel(const float* d_A, const float* d_B, float* d_C);

// 向量标量乘法
__global__ void vectorScalarMulKernel(const float* d_A, float scalar, float* d_C, int size);

// 计算向量的模 - 核函数
__global__ void vectorMagnitudeKernel(const float* d_A, float* d_C, int size);

// 计算向量的模 - 主机函数
double computeVectorMagnitude(const float* d_A, int size);

// 向量取顶
__global__ void vectorCeilKernel(const float* d_input, float* d_output, int size);

// 向量取底
__global__ void vectorFloorKernel(const float* d_input, float* d_output, int size);

// 向量四舍五入
__global__ void vectorRoundKernel(const float* d_input, float* d_output, int size);

#endif // CU_VECTOR_OPS_H

