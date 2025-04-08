//
// Created by user on 4/8/25.
//

#include "serverlet/c_yolo_v8_apis.h"

#include <opencv2/opencv.hpp>

int main() {

    // Initialize the model
    c_yolo_init("/opt/models/yolov8n.engine");

    // Load the image, and check if it was loaded successfully
    cv::Mat image = cv::imread("./test.png", cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Error: Could not load image." << std::endl;
        c_yolo_release();
        return -1;
    }

    // Load the image to the model
    c_yolo_add_image(0, image.data, 3, image.cols, image.rows);

    // Run inference
    if (!c_yolo_inference()) {
        std::cerr << "Error: Inference failed." << std::endl;
        c_yolo_release();
    }

    // Print out the counts available
    int count_results = c_yolo_available_results(0, 0.4, 0.4);
    std::cout << "Count of results: " << count_results << std::endl;

    // Get the first detected object from the model
    auto fptr_results = c_yolo_get_result(0);
    for (int i = 0; i < LEN_YOLO_ENTITY; i++) {
        std::cout << fptr_results[i] << "\t";
    }
    std::cout << std::endl;

    // Release the model
    c_yolo_release();
    return 0;
}