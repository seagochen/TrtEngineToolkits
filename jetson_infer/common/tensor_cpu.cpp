//
// Created by Orlando Chen on 8/23/24.
//
#include "tensor.h"
#include <stdexcept>


// CpuTensor 类的构造函数实现
CpuTensor::CpuTensor(const TensorDimensions& dims)
        : TensorBase(dims), data(dims.size / sizeof(float)) {}

// 从 CudaTensor 拷贝数据到 CpuTensor 的实现
void CpuTensor::copyFrom(const CudaTensor& cudaTensor) {
    if (dims.size != cudaTensor.getSize()) {
        throw std::runtime_error("Tensor sizes do not match.");
    }
    cudaMemcpy(data.data(), cudaTensor.getData(), dims.size, cudaMemcpyDeviceToHost);
}

void CpuTensor::copyFrom(const CpuTensor &other) {
    // 检查两个张量的尺寸是否匹配
    if (dims.size != other.getSize()) {
        throw std::runtime_error("Tensor sizes do not match.");
    }
    std::copy(other.getData(), other.getData() + data.size(), data.begin());
}


CudaTensor &CpuTensor::toCuda() {
    CudaTensor* cudaTensor = new CudaTensor(dims);
    cudaMemcpy(cudaTensor->getData(), this->getData(), dims.size, cudaMemcpyHostToDevice);
    return *cudaTensor;
}