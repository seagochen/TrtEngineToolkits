#ifndef __CUDA_VECTOR_HPP__
#define __CUDA_VECTOR_HPP__

#include <cuda_runtime.h>

// 向量加法
__global__ void vectorAddKernel(float* d_A, float* d_B, float* d_C, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        d_C[i] = d_A[i] + d_B[i];
    }
}

// 向量减法
__global__ void vectorSubKernel(float* d_A, float* d_B, float* d_C, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        d_C[i] = d_A[i] - d_B[i];
    }
}

// 向量点积
__global__ void vectorDotProductKernel(float* d_A, float* d_B, float* d_C, int size) {
    __shared__ float temp[256];
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;

    temp[tid] = (i < size) ? d_A[i] * d_B[i] : 0.0f;

    __syncthreads();

    // Reduction within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            temp[tid] += temp[tid + stride];
        }
        __syncthreads();
    }

    // Sum reduction from each block
    if (tid == 0) {
        atomicAdd(d_C, temp[0]);
    }
}

// 向量叉乘
__global__ void vectorCrossProductKernel(float* d_A, float* d_B, float* d_C) {
    int i = threadIdx.x;
    if (i == 0) {
        d_C[0] = d_A[1] * d_B[2] - d_A[2] * d_B[1];
    } else if (i == 1) {
        d_C[1] = d_A[2] * d_B[0] - d_A[0] * d_B[2];
    } else if (i == 2) {
        d_C[2] = d_A[0] * d_B[1] - d_A[1] * d_B[0];
    }
}

// 向量标量乘法
__global__ void vectorScalarMulKernel(float* d_A, float scalar, float* d_C, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        d_C[i] = d_A[i] * scalar;
    }
}

// 计算向量的模-核函数
__global__ void vectorMagnitudeKernel(float* d_A, float* d_C, int size) {
    __shared__ float temp[256];
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;

    temp[tid] = (i < size) ? d_A[i] * d_A[i] : 0.0f;

    __syncthreads();

    // Reduction within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            temp[tid] += temp[tid + stride];
        }
        __syncthreads();
    }

    // Sum reduction from each block
    if (tid == 0) {
        atomicAdd(d_C, temp[0]);
    }
}

// 计算向量的模
float computeVectorMagnitude(float* d_A, int size) {
    float result = 0.0f;
    float* d_result;
    cudaMalloc((void**)&d_result, sizeof(float));
    cudaMemset(d_result, 0, sizeof(float));  // 确保 d_result 初始化为 0

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    vectorMagnitudeKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_result, size);

    cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_result);

    return sqrtf(result);
}

// 向量取顶
__global__ void vectorCeilKernel(float* d_input, float* d_output, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        d_output[i] = ceilf(d_input[i]);
    }
}

// 向量取底
__global__ void vectorFloorKernel(float* d_input, float* d_output, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        d_output[i] = floorf(d_input[i]);
    }
}

// 向量四舍五入
__global__ void vectorRoundKernel(float* d_input, float* d_output, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        d_output[i] = roundf(d_input[i]);
    }
}

#endif // __CUDA_VECTOR_HPP__
