#ifndef CU_MATRIX_H
#define CU_MATRIX_H

#include <cuda_runtime.h>

// 矩阵加法
__global__ void matrixAddKernel(const float* A, const float* B, float* C, int numRows, int numCols);

// 矩阵减法
__global__ void matrixSubKernel(const float* A, const float* B, float* C, int numRows, int numCols);

// 矩阵逐元素乘法
__global__ void matrixElementwiseMulKernel(const float* A, const float* B, float* C, int numRows, int numCols);

// 矩阵逐元素除法
__global__ void matrixElementwiseDivKernel(const float* A, const float* B, float* C, int numRows, int numCols);

// 矩阵标量乘法
__global__ void matrixScalarMulKernel(const float* A, float scalar, float* C, int numRows, int numCols);

// 矩阵标量除法
__global__ void matrixScalarDivKernel(const float* A, float scalar, float* C, int numRows, int numCols);

// 矩阵标量加法
__global__ void matrixScalarAddKernel(const float* A, float scalar, float* C, int numRows, int numCols);

// 矩阵标量减法
__global__ void matrixScalarSubKernel(const float* A, float scalar, float* C, int numRows, int numCols);

// 矩阵乘法
__global__ void matrixMulKernel(const float* __restrict__ d_M, const float* __restrict__ d_N, float* d_O, int M_rows, int M_cols, int N_cols);

// 矩阵转置
__global__ void matrixTransposeKernel(const float* __restrict__ d_input, float* d_output, int rows, int cols);

// 矩阵行求和
__global__ void matrixRowSumKernel(const float* A, float* rowSums, int numRows, int numCols);

// 矩阵列求和
__global__ void matrixColSumKernel(const float* A, float* colSums, int numRows, int numCols);

// 矩阵行最大值
__global__ void matrixRowMaxKernel(const float* A, float* rowMax, int numRows, int numCols);

// 矩阵列最大值
__global__ void matrixColMaxKernel(const float* A, float* colMax, int numRows, int numCols);

// 矩阵点积
__global__ void matrixDotProductKernel(const float* A, const float* B, float* result, int numRows, int numCols);

// Frobenius 范数
__global__ void matrixFrobeniusNormKernel(const float* A, float* result, int numRows, int numCols);

// 矩阵归一化
__global__ void matrixNormalizeKernel(float* A, float* result, int numRows, int numCols, float minVal, float maxVal);

// 矩阵最大值
__global__ void findMaxKernel(const float* d_input, float* d_max, int size);

// 矩阵最小值
__global__ void findMinKernel(const float* d_input, float* d_min, int size);

#endif // CU_MATRIX_H
