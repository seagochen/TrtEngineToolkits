#ifndef JETSON_INFER_TENSOR_HPP
#define JETSON_INFER_TENSOR_HPP

#include <utility>
#include <vector>
#include <memory>
#include <iostream>
#include <cuda_runtime.h>
#include <unordered_map>
#include <numeric>
#include <stdexcept>
#include <algorithm>

////////////////////////////////////// Tensor Data Type //////////////////////////////////////

enum tensor_type {
    FLOAT32,
    FLOAT64,
    INT32,
    INT64,
    UINT8,
    // Add other data types as needed
};

////////////////////////////////////// Tensor Dim //////////////////////////////////////

struct TensorDimensions {
    std::vector<int> dims;
    size_t mem_size;  // 数据所占内存大小
    tensor_type type;

    TensorDimensions() : mem_size(0), type(FLOAT32) {} // Default constructor

    TensorDimensions(std::vector<int> dims, tensor_type type)
            : dims(std::move(dims)), type(type) {
        mem_size = calculateMemSize();
    }

    // 计算数据所占的内存大小
    size_t calculateMemSize() const {
        static const std::unordered_map<tensor_type, size_t> type_size_map = {
                {FLOAT32, sizeof(float)},
                {FLOAT64, sizeof(double)},
                {INT32, sizeof(int32_t)},
                {INT64, sizeof(int64_t)},
                {UINT8, sizeof(uint8_t)}
        };

        auto it = type_size_map.find(type);
        if (it == type_size_map.end()) {
            throw std::runtime_error("Unsupported data type.");
        }

        size_t elementSize = it->second;
        size_t totalElements = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>());

        return totalElements * elementSize;
    }
};

////////////////////////////////////// Tensor Base //////////////////////////////////////

template <typename T>
class TensorBase {
protected:
    TensorDimensions dims;

public:
    TensorBase() = default;
    explicit TensorBase(TensorDimensions dims) : dims(std::move(dims)) {}
    virtual ~TensorBase() = default;  // 虚析构函数
    [[nodiscard]] const TensorDimensions& getDims() const { return dims; }
    [[nodiscard]] size_t getMemSize() const { return dims.mem_size; }
    [[nodiscard]] size_t getElementCount() const { return dims.mem_size / sizeof(T); }
};

////////////////////////////////////// Forward Declaration //////////////////////////////////////

template <typename T>
class CpuTensor;

////////////////////////////////////// Tensor CUDA //////////////////////////////////////

template <typename T>
class CudaTensor : public TensorBase<T> {
private:
    std::unique_ptr<T, decltype(&cudaFree)> data;

public:

    // 默认构造函数
    CudaTensor() : TensorBase<T>(), data(nullptr, cudaFree) {}

    // 通过TensorDimensions构造
    explicit CudaTensor(const TensorDimensions& dims) : TensorBase<T>(dims), data(nullptr, cudaFree) {
        T* buffer;
        cudaError_t err = cudaMalloc(&buffer, dims.mem_size);
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA memory allocation failed");
        }
        data.reset(buffer);
    }


    // 从std::vector中拷贝数据
    void copyFrom(const std::vector<T>& inputData) {
        if (inputData.size() * sizeof(T) != this->dims.mem_size) {
            throw std::runtime_error("Tensor sizes do not match.");
        }
        cudaError_t err = cudaMemcpy(this->data.get(), inputData.data(), this->dims.mem_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA memcpy failed");
        }
    }


    // 从另一个CudaTensor中拷贝数据
    void copyFrom(const CudaTensor& other) {
        if (this->dims.mem_size != other.getMemSize()) {
            throw std::runtime_error("Tensor sizes do not match.");
        }
        cudaError_t err = cudaMemcpy(this->data.get(), other.getData().data(), this->dims.mem_size, cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA memcpy failed");
        }
    }


    // 从CpuTensor中拷贝数据
    void copyFrom(const CpuTensor<T>& cpuTensor) {
        if (this->dims.mem_size != cpuTensor.getMemSize()) {
            throw std::runtime_error("Tensor sizes do not match.");
        }
        cudaError_t err = cudaMemcpy(this->data.get(), cpuTensor.getData().data(), this->dims.mem_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA memcpy failed");
        }
    }


    // 将数据拷贝到std::vector中
    void copyTo(std::vector<T>& outputData) {

        // 如果cpuData数据长度和dims.mem_size不一致，需要重新分配内存
        if (outputData.size() * sizeof(T) != this->dims.mem_size) {
            outputData.resize(this->dims.mem_size / sizeof(T));
        }

        // 拷贝数据到cpuData
        cudaError_t err = cudaMemcpy(outputData.data(), data.get(), this->dims.mem_size, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA memcpy failed");
        }
    }


    // 将数据拷贝到另一个CudaTensor中
    void copyTo(CudaTensor& other) {
        if (this->dims.mem_size != other.getMemSize()) {
            throw std::runtime_error("Tensor sizes do not match.");
        }
        cudaError_t err = cudaMemcpy(other.getData().get(), data.get(), this->dims.mem_size, cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA memcpy failed");
        }
    }


    // 将数据拷贝到CpuTensor中
    void copyTo(CpuTensor<T>& cpuTensor) {
        if (this->dims.mem_size != cpuTensor.getMemSize()) {
            throw std::runtime_error("Tensor sizes do not match.");
        }
        cudaError_t err = cudaMemcpy(cpuTensor.getData().data(), data.get(), this->dims.mem_size, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA memcpy failed");
        }
    }


    // 暴露数据指针，方便引擎直接调用
    void* ptr() {
        return data.get();
    }


    // 添加显式的移动赋值操作符，并确保与默认的异常规范一致
    CudaTensor& operator=(CudaTensor&& other) noexcept {
        if (this != &other) {
            data = std::move(other.data);
            this->dims = std::move(other.dims);
        }
        return *this;
    }


    // 移动构造函数
    CudaTensor(CudaTensor&& other) noexcept = default;
};

////////////////////////////////////// Tensor CPU //////////////////////////////////////

template <typename T>
class CpuTensor : public TensorBase<T> {
private:
    std::vector<T> data;

public:

    // 默认构造函数
    CpuTensor() = default;

    // 通过TensorDimensions构造
    explicit CpuTensor(const TensorDimensions& dims)
            : TensorBase<T>(dims), data(dims.mem_size / sizeof(T)) {}

    // 实现copyFrom方法
    void copyFrom(const std::vector<T>& inputData) {
        if (inputData.size() * sizeof(T) != this->dims.mem_size) {
            throw std::runtime_error("Tensor sizes do not match.");
        }
        data = inputData;
    }

    // 从另一个CudaTensor中拷贝数据
    void copyFrom(const CudaTensor<T>& cudaTensor) {
        if (this->dims.mem_size != cudaTensor.getMemSize()) {
            throw std::runtime_error("Tensor sizes do not match.");
        }
        cudaError_t err = cudaMemcpy(data.data(), cudaTensor.getData().data(), this->dims.mem_size, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA memcpy failed");
        }
    }

    // 从CpuTensor中拷贝数据
    void copyFrom(const CpuTensor& other) {
        if (this->dims.mem_size != other.getMemSize()) {
            throw std::runtime_error("Tensor sizes do not match.");
        }
        data = other.getData();
    }

    // 将数据拷贝到std::vector中
    void copyTo(std::vector<T>& outputData) {
        outputData = data;
    }

    // 将数据拷贝到另一个CudaTensor中
    void copyTo(CudaTensor<T>& cudaTensor) {
        if (this->dims.mem_size != cudaTensor.getMemSize()) {
            throw std::runtime_error("Tensor sizes do not match.");
        }
        cudaError_t err = cudaMemcpy(cudaTensor.getData().get(), data.data(), this->dims.mem_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA memcpy failed");
        }
    }

    // 将数据拷贝到CpuTensor中
    void copyTo(CpuTensor& other) {
        if (this->dims.mem_size != other.getMemSize()) {
            throw std::runtime_error("Tensor sizes do not match.");
        }
        other.copyFrom(data);
    }

    // 添加显式的移动赋值操作符，并确保与默认的异常规范一致
    CpuTensor& operator=(CpuTensor&& other) noexcept {
        if (this != &other) {
            data = std::move(other.data);
            this->dims = std::move(other.dims);
        }
        return *this;
    }

    // 移动构造函数
    CpuTensor(CpuTensor&& other) noexcept = default;
};

#endif //JETSON_INFER_TENSOR_HPP
