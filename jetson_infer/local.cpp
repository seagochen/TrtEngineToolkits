#include <opencv2/opencv.hpp>
#include <csignal>
#include <stdexcept>
#include <iostream>
#include <filesystem>
#include <iomanip> 

#include "common/engine/InferWrapper.h"

#define MODEL_PATH "./models/yolov8n.dynamic.engine"
#define IMAGE_PATH "./images/human_and_pets.png"

struct YoloResult {
    int lx, ly, rx, ry, cls;
    float conf;
};

std::vector<YoloResult> results;

void decode(float* raw, size_t size) {

    int count = 0;
    int features = 84;
    int samples = 8400;

    for (int i = 0; i < 100; i++) {
        if (raw[i * features + 4] > 0.0) {
            YoloResult result;

            result.lx = int(raw[i * features + 0]);
            result.ly = int(raw[i * features + 1]);
            result.rx = int(raw[i * features + 2]);
            result.ry = int(raw[i * features + 3]);
            result.conf = raw[i * features + 4];
            result.cls = int(raw[i * features + 5]);

            results.push_back(result);
        }
    }
}

int main() {

    // Load the TensorRT engine from the serialized engine file
    InferWrapper infer(MODEL_PATH,
    {
        {"input", "images"},
        {"output", "output0"}
    },
    {1, 3, 640, 640},
    {1, 84, 8400});

    // Load the image
    cv::Mat image = cv::imread(IMAGE_PATH);

    // Preprocess the image
    infer.preprocess(image);

    std::cout << "Available solts: " << infer.getAvailableSlot() << std::endl;

    // Perform inference
    if (infer.getAvailableSlot() == 0) {
        infer.infer(0.5, 0.1);
    }

    // Get the results
    infer.getResults(0, decode);

    // Show the result
    cv::resize(image, image, cv::Size(640, 640));
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

   // Show image
   cv::imshow("Image", image);
   cv::waitKey(0);

    return 0;
}