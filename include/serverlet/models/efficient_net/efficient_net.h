//
// Created by user on 4/22/25.
//

#ifndef COMBINEDPROJECT_EFFICIENTNET_H
#define COMBINEDPROJECT_EFFICIENTNET_H

#include <opencv2/opencv.hpp>

#include "serverlet/models/infer_model.h"
#include "serverlet/models/efficient_net/efficient_def.h"

class EfficientNet final: public InferModelBase {

    int m_int_maximumBatch;     // Maximum number of batch

    int m_int_inputWidth;       // Input width
    int m_int_inputHeight;      // Input height
    int m_int_inputChannels;    // Input channels

    cv::Mat m_cv_resizedImg;    // Resized image for normalization
    cv::Mat m_cv_floatImg;      // Float data for normalization

    // vector of output data
    std::vector<float> m_vec_final_output_feat;     // for feature extraction
    std::vector<float> m_vec_final_output_class;    // for classification

    // vector of input temporary buffer
    std::vector<Tensor<float>> m_vec_input_buffs;

    // vector of output temporary buffer
    std::vector<Tensor<float>> m_vec_temp_output_feat;
    std::vector<Tensor<float>> m_vec_temp_output_class;

public:
    // Constructor
    explicit EfficientNet(const std::string& engine_path,
                          const std::string& input_name,
                          const std::vector<int>& input_shape,
                          const std::string& output_feat_name,
                          const std::vector<int>& output_feat_shape,
                          const std::string& output_class_name,
                          const std::vector<int>& output_class_shape);

    // Destructor
    ~EfficientNet() override;

    // Preprocess the input image
    void preprocess(const cv::Mat& image, int batchIdx) override;

    // Postprocess the output data
    std::vector<EfficientNetOutput> postprocess(int batchIdx);

private:
    static std::vector<EfficientNetOutput> decode(const std::vector<float>& vec_feat, const std::vector<float>& vec_class);
};


#endif //COMBINEDPROJECT_EFFICIENTNET_H
