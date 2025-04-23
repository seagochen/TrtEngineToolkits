//
// Created by user on 4/22/25.
//
#include "serverlet/models/efficient_net/efficient_net.h"
#include "serverlet/utils/logger.h"
#include "simple_cuda_toolkits/tsutils/convert.h"


[[maybe_unused]] EfficientNetForFeatAndClassification::EfficientNetForFeatAndClassification(
        const std::string& engine_path,
        int maximum_batch)
        : InferModelBaseMulti(
        engine_path,
        // 输入：batch×3×224×224
        std::vector<TensorDefinition>{
                {"input", {maximum_batch, 3, 224, 224}}
        },
        // 输出：logits(batch×2) + feat(batch×256)
        std::vector<TensorDefinition>{
                {"logits", {maximum_batch, 2}},
                {"feat",   {maximum_batch, 256}}
        }
),
          g_int_maximumBatch(maximum_batch),
          g_int_inputWidth(224),
          g_int_inputHeight(224),
          g_int_inputChannels(3),
          g_vec_inputData(static_cast<size_t>(g_int_inputChannels) *
                          g_int_inputHeight * g_int_inputWidth),
          g_vec_featData(256),
          g_vec_classData(2)
{
    LOG_VERBOSE("EfficientNet", "EfficientNetForFeatAndClassification created");
}

EfficientNetForFeatAndClassification::~EfficientNetForFeatAndClassification() = default;

//void EfficientNetForFeatAndClassification::preprocess(
//        const cv::Mat& image,
//        int batchIdx)
//{
//    // 如果batchIdx >= g_int_maximumBatch，则不处理数据
//    if (batchIdx >= g_int_maximumBatch) {
//        LOG_ERROR("EfficientNet", "batchIdx >= g_int_maximumBatch");
//        return;
//    }
//
//    // 1) Resize + 转 float
//    cv::Mat resized, floatImg;
//    cv::resize(image, resized,
//               cv::Size(g_int_inputWidth, g_int_inputHeight));
//    resized.convertTo(floatImg, CV_32FC3);
//
//    // 数据
//
//    // 2) HWC -> CHW
//    size_t idx = 0;
//    float mean[3] = {0.485f, 0.456f, 0.406f};
//    float std[3]  = {0.229f, 0.224f, 0.225f};
//    for (int c = 0; c < g_int_inputChannels; ++c) {
//        for (int h = 0; h < g_int_inputHeight; ++h) {
//            const float* rowPtr = floatImg.ptr<float>(h); // 使用指针访问
//            for (int w = 0; w < g_int_inputWidth; ++w) {
//                // 先将数据收缩到[0,1]
//
//
//                int index = idx++;
//                g_vec_inputData[index] = (rowPtr[w * 3 + c] - mean[c]) / std[c];
//            }
//        }
//    }
//
//    // 3) 上传到 GPU
//    copyCpuDataToInputBuffer("input", g_vec_inputData, batchIdx);
//}

void EfficientNetForFeatAndClassification::preprocess(
        const cv::Mat& image,
        int batchIdx)
{
    // 1) 边界检查
    if (batchIdx >= g_int_maximumBatch) {
        LOG_ERROR("EfficientNet", "batchIdx >= g_int_maximumBatch");
        return;
    }

    // 2) 将颜色空间转换为 RGB
    cv::Mat rgbImage;
    cv::cvtColor(image, rgbImage, cv::COLOR_BGR2RGB);


    // 3) Resize 到指定大小
    cv::Mat resized;
    cv::resize(rgbImage, resized,
               cv::Size(g_int_inputWidth, g_int_inputHeight));

    // 4) 转换为 float 并归一化到 [0,1]
    cv::Mat floatImg;
    resized.convertTo(floatImg, CV_32FC3, 1.0f / 255.0f);

    // 5) 准备标准化参数
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float stdv[3] = {0.229f, 0.224f, 0.225f};

    // 6) HWC -> CHW，并在此过程中做标准化
    //    CHW 格式下，每个通道有 H*W 个元素
    const int H = g_int_inputHeight;
    const int W = g_int_inputWidth;
    const int C = g_int_inputChannels;
    const size_t singleChannelSize = static_cast<size_t>(H) * W;

    for (int c = 0; c < C; ++c) {
        for (int h = 0; h < H; ++h) {
            // 访问第 h 行的指针
            const float* rowPtr = floatImg.ptr<float>(h);
            for (int w = 0; w < W; ++w) {
                // HWC 存储：idx = h*W*3 + w*3 + c
                float val = rowPtr[w * 3 + c];
                // 标准化
                val = (val - mean[c]) / stdv[c];
                // CHW 存储：offset = c*(H*W) + h*W + w
                size_t offset = static_cast<size_t>(c) * singleChannelSize
                                + static_cast<size_t>(h) * W
                                + static_cast<size_t>(w);
                g_vec_inputData[offset] = val;
            }
        }
    }

    // 7) 上传到 GPU
    copyCpuDataToInputBuffer("input", g_vec_inputData, batchIdx);
}


std::vector<float> EfficientNetForFeatAndClassification::postprocess(
        int batchIdx)
{
    // 如果batchIdx >= g_int_maximumBatch，则不处理数据
    if (batchIdx >= g_int_maximumBatch) {
        LOG_ERROR("EfficientNet", "batchIdx >= g_int_maximumBatch");
        return {};
    }

    // 1) 读回特征和分类 logits
    copyCpuDataFromOutputBuffer("feat",   g_vec_featData,  batchIdx);
    copyCpuDataFromOutputBuffer("logits", g_vec_classData, batchIdx);

    // 2) 找到 logits 中最大值的下标
    // auto val1 = abs(g_vec_classData[0]);
    // auto val2 = abs(g_vec_classData[1]);
    auto val1 = g_vec_classData[0];
    auto val2 = g_vec_classData[1];
    int maxIndex = (val1 > val2) ? 0 : 1;

    // auto it = std::max_element(g_vec_classData.begin(), g_vec_classData.end());
    // int maxIndex = static_cast<int>(std::distance(g_vec_classData.begin(), it));

    // 3) 构造最终返回： [maxIndex, feat0, feat1, ..., featN]
    std::vector<float> result;
    result.reserve(1 + g_vec_featData.size());
    result.push_back(static_cast<float>(maxIndex));            // result[0]
    result.insert(result.end(),                             
                  g_vec_featData.begin(), 
                  g_vec_featData.end());                     // 后面都是特征

    return result;
}

// std::vector<float> EfficientNetForFeatAndClassification::decode(
//         const std::vector<float>& vec_feat,
//         const std::vector<float>& vec_class)
// {
//     // 找到分类最大值的下标
//     auto it = std::max_element(vec_class.begin(), vec_class.end());
//     int maxIndex = static_cast<int>(std::distance(vec_class.begin(), it));

//     // std::vector<float> result;
//     // result.reserve(vec_feat.size() + vec_class.size());
//     // // 先放分类,再放特征
//     // result.insert(result.end(), vec_class.begin(), vec_class.end());
//     // result.insert(result.end(), vec_feat.begin(),  vec_feat.end());
//     // return result;

//     // 构造结果
//     std::vector<float> result;
//     result.reserve(1 + vec_feat.size());
//     result.push_back(static_cast<float>(maxIndex));
//     result.insert(result.end(), vec_feat.begin(), vec_feat.end());
//     return result;
// }
