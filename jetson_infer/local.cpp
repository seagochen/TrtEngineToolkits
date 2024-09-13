#include <opencv2/opencv.hpp>
#include <csignal>
#include <stdexcept>
#include <iostream>
#include <filesystem>

#include "common/engine/engine_loader.h"
#include "common/engine/inference.h"

#define MODEL_PATH "./models/yolov8n.dynamic.engine"
#define IMAGE_PATH "./images/human_and_pets.png"

#define INPUT_WIDTH 640
#define INPUT_HEIGHT 640
#define INPUT_CHANNELS 3

int main() {

    // Set the batch size
    int batch_size = 2;

    // Load the TensorRT engine from the file
    auto engine = loadEngineFromFile(MODEL_PATH);
    if (engine == nullptr) {
        std::cerr << "Failed to load engine from file" << std::endl;
        return 1;
    }
    std::cout << "Engine loaded successfully" << std::endl;

    // Set up the context for inference
    auto context = createExecutionContext(engine, "images", nvinfer1::Dims4(batch_size, INPUT_CHANNELS, INPUT_WIDTH, INPUT_HEIGHT));
    std::cout << "Context created successfully" << std::endl;

    // Create CUDA buffers for input and output
    std::map<std::string, std::vector<int>> tensor_dims;
    tensor_dims["images"] = {batch_size, INPUT_CHANNELS, INPUT_WIDTH, INPUT_HEIGHT};
    tensor_dims["output0"] = {batch_size, 84, 8400};
    auto buffers = allocateCudaTensors(tensor_dims);
    std::cout << "Buffers for TensorRT engine are ready" << std::endl;

    // Load the image for inference
    cv::Mat image = cv::imread(IMAGE_PATH);

    // Preprocess the image
    preprocess(image, buffers["images"]);

    // Run inference
    inference(context, buffers["images"], buffers["output0"]);
    std::cout << "Inference done" << std::endl;

    // Postprocess the results
    std::vector<YoloResult> results;
    postprocess(buffers["output0"], results, 0.1);

    std::cout << results.size() << std::endl;

    // Draw the bounding boxes
    cv::resize(image, image, cv::Size(INPUT_WIDTH, INPUT_HEIGHT));
    for (const auto& result : results) {
        auto lx = result.lx;
        auto ly = result.ly;
        auto rx = result.rx;
        auto ry = result.ry;
        auto cls = result.cls;
        auto conf = result.conf;

        cv::rectangle(image, cv::Point(lx, ly), cv::Point(rx, ry), cv::Scalar(0, 255, 0), 2);
        cv::putText(image, std::to_string(cls) + " " + std::to_string(conf), cv::Point(lx, ly), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
    }

    // Display the image
    cv::imshow("Image", image);
    cv::waitKey(0);

    return 0;
}


//int main() {
//    // Load the TensorRT engine from the file
//    auto engine = loadEngineFromFile(MODEL_PATH);
//    if (engine == nullptr) {
//        std::cerr << "Failed to load engine from file" << std::endl;
//        return 1;
//    }
//
//    std::cout << "Engine loaded successfully" << std::endl;
//
//    // Create context from the engine
//    auto context = createExecutionContext(engine, "images", nvinfer1::Dims4(1,3,640,640));
//
//    // Create CUDA buffers for input and output
//    auto buffers = loadTensorsFromModel(engine);
//
//    std::cout << "Buffers loaded successfully" << std::endl;
//
//    // Initialize the buffers for TensorRT model
//    initCudaTemporaryBuffer(640, 640);
//
//    // Load the image for inference
//    cv::Mat image = cv::imread(IMAGE_PATH);
//    if (image.empty()) {
//        std::cerr << "Failed to load image" << std::endl;
//        return 1;
//    }
//
//    // Preprocess the image
//    preprocess(image, buffers["images"]);
//
//    // Run inference
//    inference(context, buffers["images"], buffers["output0"]);
//
//    // Postprocess the output
//    std::vector<YoloResult> results;
//    postprocess(buffers["output0"], results, 0.1);
//
//    // Draw the bounding boxes
//    for (const auto& result : results) {
//        auto lx = result.lx;
//        auto ly = result.ly;
//        auto rx = result.rx;
//        auto ry = result.ry;
//        auto cls = result.cls;
//        auto conf = result.conf;
//
//        cv::rectangle(image, cv::Point(lx, ly), cv::Point(rx, ry), cv::Scalar(0, 255, 0), 2);
//        cv::putText(image, std::to_string(cls) + " " + std::to_string(conf), cv::Point(lx, ly), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
//    }
//
//    // Display the image
//    cv::imshow("Image", image);
//    cv::waitKey(0);
//
//    return 0;
//}