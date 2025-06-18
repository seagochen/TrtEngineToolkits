//
// Created by user on 6/18/25.
//

#ifndef INFER_YOLO_V8_BASE_H
#define INFER_YOLO_V8_BASE_H


#ifndef INFER_YOLO_V8_H
#define INFER_YOLO_V8_H

#include <vector>
#include "serverlet/models/infer_model_multi.h"

template<typename YoloResultType, typename ConvertFunc>
class InferYoloV8 final : public InferModelBaseMulti {
public:
    explicit InferYoloV8(const std::string& engine_path,
                         int maximum_batch,
                         int maximum_items,
                         int infer_features_val, // Pass this to constructor
                         const std::vector<TensorDefinition>& output_tensor_defs,
                         ConvertFunc converter);

    ~InferYoloV8() override;

    void preprocess(const cv::Mat& image, int batchIdx) override;

    [[nodiscard]] std::vector<YoloResultType> postprocess(int batchIdx=0, float cls=0.4, float iou=0.5);

private:
    int maximum_batch;
    int maximum_items;
    int image_width;
    int image_height;
    int image_channels;
    int infer_features;
    int infer_samples;

    std::vector<float> g_vec_output;
    ConvertFunc m_converter; // Store the conversion function
};


#endif //INFER_YOLO_V8_BASE_H
