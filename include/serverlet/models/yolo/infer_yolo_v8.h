#ifndef INFER_YOLO_V8_H
#define INFER_YOLO_V8_H

#include <vector>
#include "serverlet/models/infer_model_multi.h"
#include "serverlet/models/common/yolo_dstruct.h"


class InferYoloV8Obj final: public InferModelBaseMulti {
public:
    // Constructor and destructor
    // engine_path: TensorRT 引擎文件路径
    // maximum_batch: 最大 1<= batch <=8
    explicit InferYoloV8Obj(const std::string& engine_path, int maximum_batch = 1);

    // Destructor
    ~InferYoloV8Obj() override;

    // Preprocess the image
    void preprocess(const cv::Mat& image, int batchIdx) override;

    // Postprocess the output
    std::vector<Yolo> postprocess(int batchIdx=0, float cls=0.4);

private:
    int g_int_maximumBatch;     // Maximum number of batch
    int g_int_inputWidth;       // Input width
    int g_int_inputHeight;      // Input height
    int g_int_inputChannels;    // Input channels
    int g_int_outputFeatures;   // Number of output features
    int g_int_outputSamples;    // Number of output samples

    std::vector<float> g_vec_output; // Output buffer for postprocessing
};


class InferYoloV8Pose final: public InferModelBaseMulti {
public:
    // Constructor and destructor
    // engine_path: TensorRT 引擎文件路径
    // maximum_batch: 最大 1<= batch <=8
    explicit InferYoloV8Pose(const std::string& engine_path, int maximum_batch = 1);

    // Destructor
    ~InferYoloV8Pose() override;

    // Preprocess the image
    void preprocess(const cv::Mat& image, int batchIdx) override;

    // Postprocess the output
    std::vector<YoloPose> postprocess(int batchIdx=0, float cls=0.4);

private:
    int g_int_maximumBatch;     // Maximum number of batch
    int g_int_inputWidth;       // Input width
    int g_int_inputHeight;      // Input height
    int g_int_inputChannels;    // Input channels
    int g_int_outputFeatures;   // Number of output features
    int g_int_outputSamples;    // Number of output samples

    std::vector<float> g_vec_output; // Output buffer for postprocessing
};

#endif //INFER_YOLO_V8_H
