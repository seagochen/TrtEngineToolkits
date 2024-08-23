#include "cu_matrix.cuh"

#include <float.h>

// 矩阵加法
__global__ void matrixAddKernel(const float* A, const float* B, float* C, int numRows, int numCols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numRows && col < numCols) {
        int index = row * numCols + col;
        C[index] = A[index] + B[index];
    }
}

// 矩阵减法
__global__ void matrixSubKernel(const float* A, const float* B, float* C, int numRows, int numCols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numRows && col < numCols) {
        int index = row * numCols + col;
        C[index] = A[index] - B[index];
    }
}

// 矩阵逐元素乘法
__global__ void matrixElementwiseMulKernel(const float* A, const float* B, float* C, int numRows, int numCols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numRows && col < numCols) {
        int index = row * numCols + col;
        C[index] = A[index] * B[index];
    }
}

// 矩阵逐元素除法
__global__ void matrixElementwiseDivKernel(const float* A, const float* B, float* C, int numRows, int numCols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numRows && col < numCols) {
        int index = row * numCols + col;
        C[index] = A[index] / B[index];
    }
}

// 矩阵标量乘法
__global__ void matrixScalarMulKernel(const float* A, float scalar, float* C, int numRows, int numCols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numRows && col < numCols) {
        int index = row * numCols + col;
        C[index] = A[index] * scalar;
    }
}

// 矩阵标量除法
__global__ void matrixScalarDivKernel(const float* A, float scalar, float* C, int numRows, int numCols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numRows && col < numCols) {
        int index = row * numCols + col;
        C[index] = A[index] / scalar;
    }
}

// 矩阵标量加法
__global__ void matrixScalarAddKernel(const float* A, float scalar, float* C, int numRows, int numCols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numRows && col < numCols) {
        int index = row * numCols + col;
        C[index] = A[index] + scalar;
    }
}

// 矩阵标量减法
__global__ void matrixScalarSubKernel(const float* A, float scalar, float* C, int numRows, int numCols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numRows && col < numCols) {
        int index = row * numCols + col;
        C[index] = A[index] - scalar;
    }
}

// 矩阵乘法
__global__ void matrixMulKernel(const float* __restrict__ d_M, const float* __restrict__ d_N, float* d_O, int M_rows, int M_cols, int N_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M_rows && col < N_cols) {
        float value = 0;
        for (int k = 0; k < M_cols; ++k) {
            value += d_M[row * M_cols + k] * d_N[k * N_cols + col];
        }
        d_O[row * N_cols + col] = value;
    }
}

// 矩阵转置
__global__ void matrixTransposeKernel(const float* __restrict__ d_input, float* d_output, int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        int inputIdx = y * cols + x;
        int outputIdx = x * rows + y;
        d_output[outputIdx] = d_input[inputIdx];
    }
}

// 矩阵行求和
__global__ void matrixRowSumKernel(const float* A, float* rowSums, int numRows, int numCols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < numRows) {
        float sum = 0.0f;
        for (int col = 0; col < numCols; ++col) {
            sum += A[row * numCols + col];
        }
        rowSums[row] = sum;
    }
}

// 矩阵列求和
__global__ void matrixColSumKernel(const float* A, float* colSums, int numRows, int numCols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < numCols) {
        float sum = 0.0f;
        for (int row = 0; row < numRows; ++row) {
            sum += A[row * numCols + col];
        }
        colSums[col] = sum;
    }
}

// 矩阵行最大值
__global__ void matrixRowMaxKernel(const float* A, float* rowMax, int numRows, int numCols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < numRows) {
        float maxVal = A[row * numCols];
        for (int col = 1; col < numCols; ++col) {
            float val = A[row * numCols + col];
            if (val > maxVal) {
                maxVal = val;
            }
        }
        rowMax[row] = maxVal;
    }
}

// 矩阵列最大值
__global__ void matrixColMaxKernel(const float* A, float* colMax, int numRows, int numCols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < numCols) {
        float maxVal = A[col];
        for (int row = 1; row < numRows; ++row) {
            float val = A[row * numCols + col];
            if (val > maxVal) {
                maxVal = val;
            }
        }
        colMax[col] = maxVal;
    }
}

// 矩阵点积
__global__ void matrixDotProductKernel(const float* A, const float* B, float* result, int numRows, int numCols) {
    extern __shared__ float cache[];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIdx = threadIdx.x;

    float temp = 0.0f;
    while (tid < numRows * numCols) {
        temp += A[tid] * B[tid];
        tid += blockDim.x * gridDim.x;
    }

    cache[cacheIdx] = temp;

    __syncthreads();

    // Reduction
    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIdx < i) {
            cache[cacheIdx] += cache[cacheIdx + i];
        }
        __syncthreads();
        i /= 2;
    }

    if (cacheIdx == 0) {
        atomicAdd(result, cache[0]);
    }
}

// Frobenius 范数
__global__ void matrixFrobeniusNormKernel(const float* A, float* result, int numRows, int numCols) {
    extern __shared__ float cache[];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIdx = threadIdx.x;

    float temp = 0.0f;
    while (tid < numRows * numCols) {
        temp += A[tid] * A[tid];
        tid += blockDim.x * gridDim.x;
    }

    cache[cacheIdx] = temp;

    __syncthreads();

    // Reduction
    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIdx < i) {
            cache[cacheIdx] += cache[cacheIdx + i];
        }
        __syncthreads();
        i /= 2;
    }

    if (cacheIdx == 0) {
        atomicAdd(result, cache[0]);
    }
}

// 矩阵归一化
__global__ void matrixNormalizeKernel(float* A, float* result, int numRows, int numCols, float minVal, float maxVal) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numRows && col < numCols) {
        int index = row * numCols + col;
        result[index] = (A[index] - minVal) / (maxVal - minVal);
    }
}

// 矩阵最大值
__global__ void findMaxKernel(const float* d_input, float* d_max, int size) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        sdata[tid] = d_input[i];
    } else {
        sdata[tid] = -FLT_MAX;
    }

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_max[blockIdx.x] = sdata[0];
    }
}

// 矩阵最小值
__global__ void findMinKernel(const float* d_input, float* d_min, int size) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        sdata[tid] = d_input[i];
    } else {
        sdata[tid] = FLT_MAX;
    }

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fminf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_min[blockIdx.x] = sdata[0];
    }
}
