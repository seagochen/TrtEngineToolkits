#ifndef INFER_YOLO_V8_H
#define INFER_YOLO_V8_H

#include <vector>
#include "serverlet/models/infer_model_multi.h"
#include "serverlet/models/common/yolo_dstruct.h"


class InferYoloV8Obj final: public InferModelBaseMulti {
public:

    /**
     * @brief Constructor for InferYoloV8Obj
     * @param engine_path Path to the TensorRT engine file
     * @param maximum_batch Maximum batch size (default is 1), must be between 1 and 8
     * @param maximum_items Maximum number of items to process (default is 100)
     */
    explicit InferYoloV8Obj(const std::string& engine_path, int maximum_batch = 1, int maximum_items = 100);

    /**
     * @brief Destructor
     */
    ~InferYoloV8Obj() override;

    /**
     * @brief Preprocess the input image for inference
     * @param image Input image in cv::Mat format
     * @param batchIdx Index of the batch to which this image belongs
     */
    void preprocess(const cv::Mat& image, int batchIdx) override;

    /**
     * @brief Postprocess the output from the model
     * @param batchIdx Index of the batch to process (default is 0)
     * @param cls Confidence threshold for class detection (default is 0.4)
     * @param iou IoU threshold for non-maximum suppression (default is 0.5)
     * @return A vector of Yolo objects containing detected bounding boxes and classes
     */
    [[nodiscard]] std::vector<Yolo> postprocess(int batchIdx=0, float cls=0.4, float iou=0.5);

private:
    int maximum_batch;     // Maximum number of batch
    int maximum_items;      // Maximum number of items to process
    int image_width;       // Input width
    int image_height;      // Input height
    int image_channels;    // Input channels
    int infer_features;   // Number of output features
    int infer_samples;    // Number of output samples

    std::vector<float> g_vec_output; // Output buffer for postprocessing
};


class InferYoloV8Pose final: public InferModelBaseMulti {
public:
    // Constructor and destructor
    // engine_path: TensorRT 引擎文件路径
    // maximum_batch: 最大 1<= batch <=8

    /**
     * @brief Constructor for InferYoloV8Pose
     * @param engine_path Path to the TensorRT engine file
     * @param maximum_batch Maximum batch size (default is 1), must be between 1 and 8
     * @return An instance of InferYoloV8Pose
     */
    explicit InferYoloV8Pose(const std::string& engine_path, int maximum_batch = 1, int maximum_items = 100);

    /**
     * @brief Destructor for InferYoloV8Pose
     */
    ~InferYoloV8Pose() override;

    /**
     * @brief Preprocess the input image for inference
     * @param image Input image in cv::Mat format
     * @param batchIdx Index of the batch to which this image belongs
     */
    void preprocess(const cv::Mat& image, int batchIdx) override;

    /**
     * @brief Postprocess the output from the model
     * @param batchIdx Index of the batch to process (default is 0)
     * @param cls Confidence threshold for class detection (default is 0.4)
     * @param iou IoU threshold for non-maximum suppression (default is 0.5)
     * @return A vector of YoloPose objects containing detected bounding boxes, classes, and keypoints
     */
    [[nodiscard]] std::vector<YoloPose> postprocess(int batchIdx=0, float cls=0.4, float iou=0.5);

private:
    int maximum_batch;     // Maximum number of batch
    int maximum_items;      // Maximum number of items to process
    int image_width;       // Input width
    int image_height;      // Input height
    int image_channels;    // Input channels
    int infer_features;   // Number of output features
    int infer_samples;    // Number of output samples

    std::vector<float> g_vec_output; // Output buffer for postprocessing
};

#endif //INFER_YOLO_V8_H
