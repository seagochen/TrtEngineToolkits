#ifndef INFER_EFFICIENTNET_H
#define INFER_EFFICIENTNET_H

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "serverlet/models/infer_model_multi.h"

class EfficientNetForFeatAndClassification final : public InferModelBaseMulti {
public:
    // engine_path: TensorRT 引擎文件路径
    // maximum_batch: 最大 1<= batch <=8
    explicit EfficientNetForFeatAndClassification(const std::string& engine_path, int maximum_batch = 1);

    // 把单张 OpenCV 图像预处理并上传到 GPU
    void preprocess(const cv::Mat& image, int batchIdx) override;

    // 推理完成后读取 feature + logits，并合并返回
    std::vector<float> postprocess(int batchIdx);

private:
    int g_int_maximumBatch;
    int g_int_inputWidth;
    int g_int_inputHeight;
    int g_int_inputChannels;
};


#endif //INFER_EFFICIENTNET_H
