//
// Created by vipuser on 25-1-6.
//

#include <simple_cuda_toolkits/tensor_utils.hpp>

#include <vector>
#include "serverlet/utils/logger.h"
#include "serverlet/models/yolo/infer_yolo_v8.h"
#include "serverlet/models/common/nms.hpp"
#include "serverlet/models/yolo/yolo_post_proc.h"
#include "serverlet/models/common/image_to_tensor.h"


InferYoloV8Pose::InferYoloV8Pose(
        const std::string& engine_path,
        int maximum_batch,
        int maximum_items): InferModelBaseMulti(engine_path,
            std::vector<TensorDefinition> {{"images", {maximum_batch, 3, 640, 640}}},
            std::vector<TensorDefinition> {{"output0", {maximum_batch, 56, 8400}}}) {

    image_width = 640;
    image_height = 640;
    image_channels = 3;
    maximum_batch = maximum_batch;
    infer_features = 56;
    infer_samples = 8400;

    // Initialize the output buffer
    g_vec_output.resize(infer_features * infer_samples, 0.0f);
}


InferYoloV8Pose::~InferYoloV8Pose() {

    // Release local buffer for output
    g_vec_output.clear();
    LOG_VERBOSE_TOPIC("InferYolov8Pose", "deconstructor", "Local buffer released successfully.");
}


// Preprocess the image
void InferYoloV8Pose::preprocess(const cv::Mat& image, const int batchIdx) {
    // 1) 边界检查
    if (batchIdx >= maximum_batch) {
        LOG_ERROR("EfficientNet", "batchIdx >= g_int_maximumBatch");
        return;
    }

    // 2) 标准化参数
    std::vector<float> mean = {0.485f, 0.456f, 0.406f};
    std::vector<float> stdv = {0.229f, 0.224f, 0.225f};

    // 3) 查询本次要使用的 CUDA buffer
    const float* cuda_device_ptr = accessCudaBufByBatchIdx("images", batchIdx);

    // 4) 将 const float* cuda_buffer 转换为 float* 类型
    auto cuda_buffer_float = const_cast<float*>(cuda_device_ptr);
    if (cuda_buffer_float == nullptr) {
        LOG_ERROR("EfficientNet", "Failed to access CUDA buffer for input");
        return;
    }

    // 4) 转换图片并拷贝到CUDA设备中
    sct_image_to_cuda_tensor(
        image,                  // 输入图像
        cuda_buffer_float,      // CUDA 设备指针
        image_height,      // 目标高度, image.dim0
        image_width,       // 目标宽度, image.dim1
        image_channels,    // 目标通道数, image.dim2
        false                   // 不进行 BGR 到 RGB 的转换
        );
}


// Postprocess the output
std::vector<YoloPose> InferYoloV8Pose::postprocess(const int batchIdx, const float cls, const float iou) {

    // 1) 边界检查
    if (batchIdx >= maximum_batch) {
        LOG_ERROR("EfficientNet", "batchIdx >= g_int_maximumBatch");
        return {};
    }

    // 2) 查询本次要使用的 CUDA buffer
    const float* cuda_device_ptr = accessCudaBufByBatchIdx("output0", batchIdx);

    // 3) 使用 sct_yolo_post_proc 处理输出
    int results = sct_yolo_post_proc(cuda_device_ptr, g_vec_output, infer_features, infer_samples, cls, true);
    if (results < 0) {
        LOG_ERROR("InferYoloV8Pose", "No results found after post-processing");
        return {};
    }

   // 4) 将结果转换为 Yolo 对象
    std::vector<YoloPose> yolo_results;
    host_xywh_to_xyxy_pose(g_vec_output, yolo_results, infer_features, results);

    // 5) 执行NMS处理
    yolo_results = nms(yolo_results, iou);
    return yolo_results;
}
