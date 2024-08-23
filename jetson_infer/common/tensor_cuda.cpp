//
// Created by Orlando Chen on 8/23/24.
//
#include "tensor.h"
#include <stdexcept>

// CudaTensor 类的构造函数实现
CudaTensor::CudaTensor(TensorDimensions const& dims)
        : TensorBase(dims), data(nullptr, cudaFree) {
    void* buffer;
    cudaMalloc(&buffer, dims.size);
    data.reset(buffer);
}

// 从另一个 CudaTensor 拷贝数据的实现
void CudaTensor::copyFrom(const CudaTensor& other) {
    if (dims.size != other.getSize()) {
        throw std::runtime_error("Tensor sizes do not match.");
    }
    cudaMemcpy(data.get(), other.getData(), dims.size, cudaMemcpyDeviceToDevice);
}

// 从 CpuTensor 拷贝数据到 CudaTensor 的实现
void CudaTensor::copyFrom(const CpuTensor& cpuTensor) {
    if (dims.size != cpuTensor.getSize()) {
        throw std::runtime_error("Tensor sizes do not match.");
    }
    cudaMemcpy(data.get(), cpuTensor.getData(), dims.size, cudaMemcpyHostToDevice);
}


CpuTensor &CudaTensor::toCpu() {
    // 将 CudaTensor 转换为 CpuTensor
    CpuTensor* cpuTensor = new CpuTensor(dims);
    cudaMemcpy(cpuTensor->getData(), this->getData(), dims.size, cudaMemcpyDeviceToHost);
    return *cpuTensor;
}
