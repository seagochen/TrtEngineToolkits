//
// Created by Orlando on 8/23/24.
//

#ifndef JETSON_INFER_TENSOR_UTILS_H
#define JETSON_INFER_TENSOR_UTILS_H

#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <string>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstddef>

// Tensor類型の変換：int8 ➝ float32
size_t convertInt8ToFloat32(const std::vector<int8_t>& src, std::vector<float>& dst) {
    dst.resize(src.size());
    std::transform(src.begin(), src.end(), dst.begin(), [](int8_t val) {
        return static_cast<float>(val);
    });
    return dst.size();
}

// Tensor類型の変換：float32 ➝ int8
size_t convertFloat32ToInt8(const std::vector<float>& src, std::vector<int8_t>& dst) {
    dst.resize(src.size());
    std::transform(src.begin(), src.end(), dst.begin(), [](float val) {
        if (std::isnan(val)) return int8_t(0); // NaN 转换为 0
        return static_cast<int8_t>(std::clamp(val, -128.0f, 127.0f));
    });
    return dst.size();
}

// Tensor類型の変換：float32 ➝ int32
size_t convertFloat32ToInt32(const std::vector<float>& src, std::vector<int32_t>& dst) {
    dst.resize(src.size());
    std::transform(src.begin(), src.end(), dst.begin(), [](float val) {
        if (std::isnan(val)) return int32_t(0); // NaN 转换为 0
        return static_cast<int32_t>(val);
    });
    return dst.size();
}

// Tensor類型の変換：int32 ➝ float32
size_t convertInt32ToFloat32(const std::vector<int32_t>& src, std::vector<float>& dst) {
    dst.resize(src.size());
    std::transform(src.begin(), src.end(), dst.begin(), [](int32_t val) {
        return static_cast<float>(val);
    });
    return dst.size();
}

// Tensor類型の変換：int8 ➝ float64
size_t convertInt8ToFloat64(const std::vector<int8_t>& src, std::vector<double>& dst) {
    dst.resize(src.size());
    std::transform(src.begin(), src.end(), dst.begin(), [](int8_t val) {
        return static_cast<double>(val);
    });
    return dst.size();
}

// Tensor類型の変換：float64 ➝ int8
size_t convertFloat64ToInt8(const std::vector<double>& src, std::vector<int8_t>& dst) {
    dst.resize(src.size());
    std::transform(src.begin(), src.end(), dst.begin(), [](double val) {
        if (std::isnan(val)) return int8_t(0); // NaN 转换为 0
        return static_cast<int8_t>(std::clamp(val, -128.0, 127.0));
    });
    return dst.size();
}

// Tensor類型の変換：uint8 ➝ float32
size_t convertUint8ToFloat32(const std::vector<uint8_t>& src, std::vector<float>& dst) {
    dst.resize(src.size());
    std::transform(src.begin(), src.end(), dst.begin(), [](uint8_t val) {
        return static_cast<float>(val);
    });
    return dst.size();
}

// Tensor類型の変換：float32 ➝ uint8
size_t convertFloat32ToUint8(const std::vector<float>& src, std::vector<uint8_t>& dst) {
    dst.resize(src.size());
    std::transform(src.begin(), src.end(), dst.begin(), [](float val) {
        if (std::isnan(val)) return uint8_t(0); // NaN 转换为 0
        return static_cast<uint8_t>(std::clamp(val, 0.0f, 255.0f));
    });
    return dst.size();
}

// Tensor類型の変換：uint8 ➝ float64
size_t convertUint8ToFloat64(const std::vector<uint8_t>& src, std::vector<double>& dst) {
    dst.resize(src.size());
    std::transform(src.begin(), src.end(), dst.begin(), [](uint8_t val) {
        return static_cast<double>(val);
    });
    return dst.size();
}

// Tensor類型の変換：float64 ➝ uint8
size_t convertFloat64ToUint8(const std::vector<double>& src, std::vector<uint8_t>& dst) {
    dst.resize(src.size());
    std::transform(src.begin(), src.end(), dst.begin(), [](double val) {
        if (std::isnan(val)) return uint8_t(0); // NaN 转换为 0
        return static_cast<uint8_t>(std::clamp(val, 0.0, 255.0));
    });
    return dst.size();
}


// 2 つのデータを比較する
template <typename T>
bool compareData(const std::vector<T>& data1, const std::vector<T>& data2, const float epsilon = 1e-5) {
    if (data1.size() != data2.size()) {
        std::cerr << "Data size mismatch: " << data1.size() << " != " << data2.size() << std::endl;
        return false;
    }

    for (size_t i = 0; i < data1.size(); ++i) {
        if (std::abs(data1[i] - data2[i]) > epsilon) {
            std::cerr << "Data mismatch at index " << i << ": " << data1[i] << " != " << data2[i] << std::endl;
            return false;
        }
    }

    return true;
}

#endif //JETSON_INFER_TENSOR_UTILS_H
