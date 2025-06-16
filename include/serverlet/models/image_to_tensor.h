//
// Created by user on 6/13/25.
//

#ifndef IMAGE_TO_TENSOR_H
#define IMAGE_TO_TENSOR_H

#include <opencv2/opencv.hpp>
#include <vector>

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
void sct_img_to_tensor(cv::Mat image,  float* device_ptr, std::vector<int> target_dims, std::vector<float>& mean, std::vector<float>& stdv, bool is_rgb);

#endif //IMAGE_TO_TENSOR_H
