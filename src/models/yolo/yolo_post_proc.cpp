//
// Created by user on 6/16/25.
//

#include "serverlet/models/yolo/yolo_post_proc.h"
#include "serverlet/models/common/nms.hpp"

#include  <cuda_runtime.h>

#include <simple_cuda_toolkits/tsutils/filter.h>
#include <simple_cuda_toolkits/tsutils/sort.h>
#include <simple_cuda_toolkits/matrix/matrix.h>

#include "serverlet/models/efficient_net/infer_efficient_net.h"


#define DEBUG 0


int sct_yolo_post_proc(const float* ptr_device, std::vector<float>& output,
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

#if DEBUG
    sctDumpCudaMemoryToCSV(ptr_device_temp0, "before_op.csv", features, samples);
#endif

    // 3. 对 ptr_device_temp0 进行转置矩阵操作
    // [1, features, samples] -> [1, samples, features]
    sctMatrixTranspose(ptr_device_temp0, ptr_device_temp1, features, samples);

#if DEBUG
    sctDumpCudaMemoryToCSV(ptr_device_temp1, "transpose.csv", samples, features);
#endif

    // 4. 当 use_pose 为 false 时，执行分类处理
    if (!use_pose)
    {
        // 算出每个样本的cls index，并把其分类id放在#5，概率放在#4（满足后续YOLO的其他模块处理）
        // sctArgmax(ptr_device_temp1, ptr_device_temp0, samples, features, 4, features, 5, 4);

        // 为了统一后续处理，将 ptr_device_temp0 的结果拷贝到 ptr_device_temp1
        std::swap(ptr_device_temp0, ptr_device_temp1); // 交换指针，速度更快

#if DEBUG
        sctDumpCudaMemoryToCSV(ptr_device_temp1, "argmax.csv", samples, features);
#endif
    }

    // 5. 对计算结果进行抑制
    int results = sctFilterGreater_dim1(
        ptr_device_temp1,
        ptr_device_temp0,
        4,
        cls,
        samples,
        features
        );

#if DEBUG
    sctDumpCudaMemoryToCSV(ptr_device_temp0, "filter.csv", samples, features);
#endif

    if (results > 0)
    {
        // 6. 执行排序作业
        sctSortTensor_dim1_descending(ptr_device_temp0, ptr_device_temp1, samples, features, 4);

#if DEBUG
        sctDumpCudaMemoryToCSV(ptr_device_temp1, "sort.csv", samples, features);
#endif

        // 执行拷贝过程 device -> host
        cudaMemcpy(output.data(), ptr_device_temp1, total_size, cudaMemcpyDeviceToHost);
    } else
    {
        results = -1; // 如果没有结果，设置为-1
    }

    // 0. 释放缓存
    cudaFree(ptr_device_temp0);
    cudaFree(ptr_device_temp1);

    // 返回结果数量
    return results;
};


