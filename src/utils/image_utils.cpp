//
// Created by user on 6/13/25.
//

#include "serverlet/utils/image_utils.h"
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <vector>

#include <simple_cuda_toolkits/tsutils/permute_3D.h>
#include <simple_cuda_toolkits/vision/normalization.h>


/**
 * @brief 将 OpenCV 图像转换为 CUDA 张量格式
 * 
 * @param image 输入的 OpenCV 图像
 * @param device_ptr CUDA 设备指针，用于存储转换后的张量数据，转换后数据格式为 [C, H, W]
 * @param target_dims 目标张量的维度，格式为 [H, W, C]，其中 C 是通道数，H 是高度，W 是宽度
 * @param mean 每个通道的均值，用于归一化
 * @param stdv 每个通道的标准差，用于归一化
 * @param is_rgb 是否将图像转换为 RGB 格式（默认是 BGR 格式）
 */
void cvtImgToCudaTensor(cv::Mat image, float* device_ptr, std::vector<int> target_dims, std::vector<float>& mean, std::vector<float>& stdv, bool is_rgb) {

    // 1. 创建一个与目标相同的CUDA Memory
    float* cuda_temp_tensor0 = nullptr;
    float* cuda_temp_tensor1 = nullptr;
    size_t total_size = target_dims[0] * target_dims[1] * target_dims[2];

    cudaMalloc(&cuda_temp_tensor0, total_size * sizeof(float));
    if (cuda_temp_tensor0 == nullptr) {
        throw std::runtime_error("Failed to allocate CUDA memory for tensor.");
    }

    cudaMalloc(&cuda_temp_tensor1, total_size * sizeof(float));
    if (cuda_temp_tensor1 == nullptr) {
        throw std::runtime_error("Failed to allocate CUDA memory for tensor.");
    }

    // 2. 将图片Resize到目标尺寸
    cv::Mat resized;
    if (image.empty()) {
        throw std::runtime_error("Input image is empty.");
    }
    auto height = target_dims[0]; // 目标高度
    auto width = target_dims[1];  // 目标宽度
    auto channels = target_dims[2]; // 目标通道数
    cv::resize(image, resized, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
    if (resized.empty()) {
        throw std::runtime_error("Failed to resize image.");
    }
    // 检查通道数是否匹配
    if (resized.channels() != channels) {
        throw std::runtime_error("Resized image channel count does not match target dimensions.");
    }

    // 2.5. 如果 is_rgb 为 true，则将 BGR 转换为 RGB
    if (is_rgb && channels == 3) {
        cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    }

    // 3. 将图像转换为float类型并归一化到[0, 1]
    cv::Mat floatImg;
    resized.convertTo(floatImg, CV_32FC3, 1.0f / 255.0f);

    // 4. 将图片拷贝到CUDA内存中
    cudaMemcpy(cuda_temp_tensor0, floatImg.data, total_size * sizeof(float), cudaMemcpyHostToDevice);

    // 5. 使用 sctNormalizeMeanStd 函数进行归一化
    sctNormalizeMeanStd(cuda_temp_tensor0, cuda_temp_tensor1, target_dims[0], target_dims[1], target_dims[2], mean.data(), stdv.data());

    // 6. 使用 sctPermute3D_v2 函数将 HWC 格式转换为 CHW 格式
    sctPermute3D_v2(cuda_temp_tensor1, cuda_temp_tensor0, target_dims[0], target_dims[1], target_dims[2], 2, 0, 1);

    // 7. 将结果拷贝到设备指针
    // 注意：这里的 device_ptr 应该是已经分配好的 CUDA 内存
    if (device_ptr == nullptr) {
        throw std::runtime_error("Device pointer is null, please ensure it is allocated before calling this function.");
    }
    cudaMemcpy(device_ptr, cuda_temp_tensor0, total_size * sizeof(float), cudaMemcpyDeviceToDevice);

    // 8. 释放临时CUDA内存
    cudaFree(cuda_temp_tensor0);
    cudaFree(cuda_temp_tensor1);
}