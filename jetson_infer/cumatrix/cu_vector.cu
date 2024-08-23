#include "cu_vector.cuh"
#include <math.h>

// 向量求和
__global__ void sumKernel(const float* d_input, float* d_sum, int size) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < size) ? d_input[i] : 0.0f;

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_sum[blockIdx.x] = sdata[0];
    }
}

// 向量加法
__global__ void vectorAddKernel(const float* d_A, const float* d_B, float* d_C, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        d_C[i] = d_A[i] + d_B[i];
    }
}

// 向量减法
__global__ void vectorSubKernel(const float* d_A, const float* d_B, float* d_C, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        d_C[i] = d_A[i] - d_B[i];
    }
}

// 向量点积
__global__ void vectorDotProductKernel(const float* d_A, const float* d_B, float* d_C, int size) {
    extern __shared__ float temp[];
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;

    temp[tid] = (i < size) ? d_A[i] * d_B[i] : 0.0f;

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            temp[tid] += temp[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(d_C, temp[0]);
    }
}

// 向量叉乘
__global__ void vectorCrossProductKernel(const float* d_A, const float* d_B, float* d_C) {
    d_C[0] = d_A[1] * d_B[2] - d_A[2] * d_B[1];
    d_C[1] = d_A[2] * d_B[0] - d_A[0] * d_B[2];
    d_C[2] = d_A[0] * d_B[1] - d_A[1] * d_B[0];
}

// 向量标量乘法
__global__ void vectorScalarMulKernel(const float* d_A, float scalar, float* d_C, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        d_C[i] = d_A[i] * scalar;
    }
}

// 计算向量的模-核函数
__global__ void vectorMagnitudeKernel(const float* d_A, float* d_C, int size) {
    extern __shared__ float temp[];
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;

    temp[tid] = (i < size) ? d_A[i] * d_A[i] : 0.0f;

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            temp[tid] += temp[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(d_C, temp[0]);
    }
}

// 计算向量的模
double computeVectorMagnitude(const float* d_A, int size) {
    float result = 0.0f;
    float* d_result;
    cudaError_t err;

    err = cudaMalloc((void**)&d_result, sizeof(float));
    if (err != cudaSuccess) {
        // 处理错误
        return -1.0;
    }
    cudaMemset(d_result, 0, sizeof(float));  // 确保 d_result 初始化为 0

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    vectorMagnitudeKernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_A, d_result, size);

    err = cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        // 处理错误
        cudaFree(d_result);
        return -1.0;
    }

    cudaFree(d_result);

    return sqrt(result);
}

// 向量取顶
__global__ void vectorCeilKernel(const float* d_input, float* d_output, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        d_output[i] = ceilf(d_input[i]);
    }
}

// 向量取底
__global__ void vectorFloorKernel(const float* d_input, float* d_output, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        d_output[i] = floorf(d_input[i]);
    }
}

// 向量四舍五入
__global__ void vectorRoundKernel(const float* d_input, float* d_output, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        d_output[i] = roundf(d_input[i]);
    }
}
