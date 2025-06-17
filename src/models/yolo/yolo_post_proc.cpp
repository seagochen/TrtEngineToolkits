//
// Created by user on 6/16/25.
//

#include "serverlet/models/yolo/yolo_post_proc.h"
#include "serverlet/models/common/nms.hpp"

#include  <cuda_runtime.h>

#include <simple_cuda_toolkits/tsutils/filter.h>
#include <simple_cuda_toolkits/tsutils/maxmin.h>
#include <simple_cuda_toolkits/matrix/matrix.h>
#include <simple_cuda_toolkits/tensor_utils.hpp>

#include "serverlet/models/efficient_net/infer_efficient_net.h"
#include "serverlet/utils/logger.h"


int sct_yolo_post_proc(const float* ptr_device, std::vector<float> output,
                                 int features, int samples, float cls, bool use_pose)
{

    // 1. 创建两个与目标相同的CUDA缓存
    float* ptr_device_temp0 = nullptr;
    float* ptr_device_temp1 = nullptr;
    size_t total_size = features * samples * sizeof(float);

    cudaMalloc(&ptr_device_temp0, total_size);
    if (ptr_device_temp0 == nullptr) {
        throw std::runtime_error("Failed to allocate CUDA memory for temp tensor 0.");
    }

    cudaMalloc(&ptr_device_temp1, total_size);
    if (ptr_device_temp1 == nullptr) {
        cudaFree(ptr_device_temp0);
        throw std::runtime_error("Failed to allocate CUDA memory for temp tensor 1.");
    }

    // 2. 将 ptr_device 的数据拷贝到 ptr_device_temp0
    cudaMemcpy(ptr_device_temp0, ptr_device, total_size, cudaMemcpyDeviceToDevice);

    // 3. 对 ptr_device_temp0 进行转置矩阵操作
    // [1, features, samples] -> [1, samples, features]
    sctMatrixTranspose(ptr_device_temp0, ptr_device_temp1, features, samples);

    // 4. 当 use_pose 为 false 时，执行分类处理
    if (!use_pose)
    {
        // 算出每个样本的cls index，并把其分类id放在#5，概率放在#4（满足后续YOLO的其他模块处理）
        sctArgmax(ptr_device_temp1, ptr_device_temp0, features, samples, 4, features, 5, 4);

        // 为了统一后续处理，将 ptr_device_temp0 的结果拷贝到 ptr_device_temp1
        // sctCudaMemcpyDtoD(ptr_device_temp0, ptr_device_temp1, total_size);
        std::swap(ptr_device_temp0, ptr_device_temp1); // 交换指针，速度更快
    }

    // 5. 对计算结果进行抑制
    int results = sctFilterAndCompactResults(ptr_device_temp1, ptr_device_temp0, 4, SCT_FILTER_GREATER_THAN,
        cls, 0.0f, features, samples);

    if (results < 0)
    {
        // 没有满足条件的结果，直接释放缓存并返回
        cudaFree(ptr_device_temp0);
        cudaFree(ptr_device_temp1);

        // 输出debug消息
        LOG_INFO("infer_postprocess", "c_yolo_inference: No results found after filtering with cls threshold");

        // 结束后续处理
        return -1;
    }

    // 6. 将结果拷贝回本地，现在结果已经全部存储在 ptr_device_temp0 中
    // 先确定 output 的大小是否 >= results * features, 如果 use_pose 为 false，则 features 为 6，否则为 features
    int output_feats = use_pose ? features : 6; // 6 是 YOLO 的标准输出格式
    if (output.size() < results * output_feats)
    {
        // 重新分配空间
        output.resize(results * output_feats);
    }

    // 执行拷贝过程 device -> host
    sctCudaMemcpyDtoH(ptr_device_temp0, output.data(), results * output_feats * sizeof(float));

    // 0. 释放缓存
    cudaFree(ptr_device_temp0);
    cudaFree(ptr_device_temp1);

    // 返回结果数量
    return results;
};
