//
// Created by user on 6/13/25.
//

#include "serverlet/models/yolo/infer_yolo_v8.h"
#include <opencv2/opencv.hpp>

int main()
{
    // Load the YOLO model
    auto model = InferYoloV8Pose("/opt/models/yolov8s-pose.engine", 4);

    // Load an image
    cv::Mat image1 = cv::imread("/opt/images/apples.png");
    cv::Mat image2 = cv::imread("/opt/images/cartoon.png");
    cv::Mat image3 = cv::imread("/opt/images/human_and_pets.png");
    cv::Mat image4 = cv::imread("/opt/images/soccer.png");

    // Verify the images were loaded successfully
    if (image1.empty() || image2.empty() || image3.empty() || image4.empty()) {
        std::cerr << "Error loading images." << std::endl;
        if (image1.empty()) {
            std::cerr << "Image 1 not found." << std::endl;
        }
        if (image2.empty()) {
            std::cerr << "Image 2 not found." << std::endl;
        }
        if (image3.empty()) {
            std::cerr << "Image 3 not found." << std::endl;
        }
        if (image4.empty()) {
            std::cerr << "Image 4 not found." << std::endl;
        }
        return -1;
    }

    // Perform inference on the images
    model.preprocess(image1, 0);
    model.preprocess(image2, 1);
    model.preprocess(image3, 2);
    model.preprocess(image4, 3);

    // Run inference
    model.inference();

    // Get the results
    auto res1 = model.postprocess(0, 0.3, 0.2);
    auto res2 = model.postprocess(1, 0.3, 0.2);
    auto res3 = model.postprocess(2, 0.3, 0.2);
    auto res4 = model.postprocess(3, 0.3, 0.2);

    std::cout << "Results for image 1: " << res1.size() << " detections." << std::endl;
    std::cout << "Results for image 2: " << res2.size() << " detections." << std::endl;
    std::cout << "Results for image 3: " << res3.size() << " detections." << std::endl;
    std::cout << "Results for image 4: " << res4.size() << " detections." << std::endl;

    return 0;
}