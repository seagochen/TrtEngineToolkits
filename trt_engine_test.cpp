//
// Created by user on 6/13/25.
//

#include <opencv2/opencv.hpp>
#include "serverlet/models/yolo/infer_yolo_v8.h"
#include "serverlet/models/common/vision_drawer.h"


int main()
{

    // Load the YOLO model
    auto model = InferYoloV8Pose("/opt/models/yolov8s-pose.engine", 4);

    // Load an image
    cv::Mat image3 = cv::imread("/opt/images/human_and_pets.png");


    // Perform inference on the images
    model.preprocess(image3, 0);

    // Run inference
    model.inference();

    // Get the results
    auto res1 = model.postprocess(0, 0.3, 0.2);

    // Print out the results
    for (const auto& item : res1) {
        std::cout << "Left: (" << item.lx << ", " << item.ly << "), "
                  << "Right: (" << item.rx << ", " << item.ry << "), "
                  << "Confidence: " << item.conf << ", "
                  << "Class: " << item.cls << std::endl;
    }

    return 0;

}