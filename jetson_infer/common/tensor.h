//
// Created by vipuser on 8/23/24.
//

#ifndef JETSON_INFER_TENSOR_H
#define JETSON_INFER_TENSOR_H

#include <utility>
#include <vector>
#include <memory>
#include <iostream>
#include <cuda_runtime.h>

////////////////////////////////////// Tensor Dim //////////////////////////////////////

struct TensorDimensions {
    std::vector<int> dims;
    size_t size;
};

////////////////////////////////////// Tensor Base //////////////////////////////////////

class TensorBase {
protected:
    TensorDimensions dims;

public:
    explicit TensorBase(TensorDimensions  dims) : dims(std::move(dims)) {}
    [[nodiscard]] const TensorDimensions& getDims() const { return dims; }
    [[nodiscard]] size_t getSize() const { return dims.size; }
};


////////////////////////////////////// Tensor CUDA //////////////////////////////////////

// 前向声明 CpuTensor 类
class CpuTensor;

class CudaTensor : public TensorBase {
private:
    std::unique_ptr<void, decltype(&cudaFree)> data;

public:
    // 默认构造函数
    CudaTensor() : TensorBase({}), data(nullptr, cudaFree) {}

    // 其他构造函数和成员函数
    explicit CudaTensor(TensorDimensions const& dims);

    inline void* getData() { return data.get(); }
    [[nodiscard]] inline const void* getData() const { return data.get(); }

    // 从另一个CudaTensor拷贝数据
    void copyFrom(const CudaTensor& other);

    // 从CpuTensor拷贝数据
    void copyFrom(const CpuTensor& cpuTensor);

    // 将CudaTensor转换为CpuTensor
    CpuTensor& toCpu();

    // 移动构造函数和移动赋值操作符
    CudaTensor(CudaTensor&& other) noexcept = default;
    CudaTensor& operator=(CudaTensor&& other) noexcept = default;
};


////////////////////////////////////// Tensor CPU //////////////////////////////////////

class CpuTensor : public TensorBase {
private:
    std::vector<float> data; // 使用std::vector来管理内存

public:
    explicit CpuTensor(const TensorDimensions& dims);

    float* getData() { return data.data(); }
    [[nodiscard]] const float* getData() const { return data.data(); }

    // 从另一个CpuTensor拷贝数据
    void copyFrom(const CpuTensor& other);

    // 从CudaTensor拷贝数据
    void copyFrom(const CudaTensor& cudaTensor);

    // 将CpuTensor转换为CUDA
    CudaTensor& toCuda();

    // 移动构造函数和移动赋值操作符
    CpuTensor(CpuTensor&& other) noexcept = default;
    CpuTensor& operator=(CpuTensor&& other) noexcept = default;
};

#endif //JETSON_INFER_TENSOR_H
