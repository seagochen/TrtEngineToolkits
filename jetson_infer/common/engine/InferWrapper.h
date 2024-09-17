//
// Created by orlando on 9/20/24.
//

#ifndef INFERWRAPPER_H
#define INFERWRAPPER_H

#include <string>
#include <memory>
#include <map>
#include <vector>

#include <opencv2/opencv.hpp>
#include <simple_cuda_toolkits/tensor.hpp>

#include "engine_loader.h"

#define MAX_BATCH_SIZE 4

class InferWrapper {

    // Engines and contexts for TensorRT
    ICudaEngineUniquePtr engine;
    IExecutionContextUniquePtr context;

    // Tensor names for input and output
    std::map<std::string, std::string> tensor_names;

    // Input and output buffers for CUDA and TensorRT
    std::map<std::string, Tensor<float>> trt_buffers;
    std::map<int, Tensor<float>> cuda_input_buffers;
    std::map<int, Tensor<float>> cuda_output_buffers;

    // The storing buffer for processed results, which are ready to be copied to the CPU
    std::vector<Tensor<float>> results;
    std::vector<float> raw_output;

    // Input and output temporary images for OpenCV
    std::map<std::string, cv::Mat> temp_images;

    // Dimensions of input and output tensors
    std::vector<int> input_dims;
    std::vector<int> output_dims;

    // Index for counting the preprocessed images and available output results
    int image_idx = 0;
    int boxes;

public:
    InferWrapper(const std::string &engine_path,            // File path for loading the engine file
        const std::map<std::string, std::string> &names,    // Names of input and output tensors
        const nvinfer1::Dims4 &input_dims,                  // Dimensions of input tensor
        const nvinfer1::Dims3 &output_dims,                 // Dimensions of output tensor
        const int boxes=1024);                              // Number of boxes for detection

    ~InferWrapper();

    /**   
    * @brief Preprocess input image for inference
    * @param image Input image for inference
    */
    void preprocess(const cv::Mat &image);

    /**
     * @brief Perform inference on the input images
     * @param images Vector of input images for inference
     */
    void preprocess(const std::vector<cv::Mat> &images);

    /**
     * @brief Perform inference on the input images
     */
    void infer(float cls_threshold, float nms_threshold, float alpha=0.f, float beta=640.f);

    /**
     * @brief Get the available slots count for storing preprocessed images
     * @return Number of available slots
     */
    int getAvailableSlot() const;

    /**
     * @brief When the inference is done, get the results
     * @return Vector of results
     */
    void getResults(int idx, void(*callback)(float*, size_t));
};

#endif //INFERWRAPPER_H