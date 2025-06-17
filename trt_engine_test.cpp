#include <opencv2/opencv.hpp>
#include "serverlet/models/yolo/infer_yolo_v8.h" // Assuming this path is correct for your model
#include "serverlet/models/common/vision_drawer.h" // Assuming this path is correct for your VisionDrawer class

#include <iostream> // For std::cerr and std::cout
#include <vector>   // For std::vector
#include <string>   // For std::string
#include <map>      // For std::map

int main()
{
    // Load the YOLO model (assuming InferYoloV8Pose uses your Yolo/YoloPose structs)
    // Make sure your InferYoloV8Pose returns std::vector<YoloPose>
    auto model = InferYoloV8Pose("/opt/models/yolov8s-pose.engine", 4);

    // Load images
    std::vector<cv::Mat> original_images;
    original_images.push_back(cv::imread("/opt/images/apples.png"));
    original_images.push_back(cv::imread("/opt/images/cartoon.png"));
    original_images.push_back(cv::imread("/opt/images/human_and_pets.png"));
    original_images.push_back(cv::imread("/opt/images/soccer.png"));

    // Verify images were loaded successfully
    std::vector<std::string> image_names = {"apples.png", "cartoon.png", "human_and_pets.png", "soccer.png"};
    for (size_t i = 0; i < original_images.size(); ++i) {
        if (original_images[i].empty()) {
            std::cerr << "Error loading image: " << image_names[i] << std::endl;
            return -1; // Exit if any image fails to load
        }
    }

    // Perform inference on the images and store results
    std::vector<std::vector<YoloPose>> all_results(original_images.size());

    for (size_t i = 0; i < original_images.size(); ++i) {
        // Preprocess (if your model requires separate preprocess calls per image/batch index)
        model.preprocess(original_images[i], i);
    }

    // Run inference once for the batch
    model.inference();

    // Get the results for each image
    for (size_t i = 0; i < original_images.size(); ++i) {
        all_results[i] = model.postprocess(i, 0.3, 0.2); // Use postprocess_idx to get results for specific image
        // Resize original image to 640x640 for consistency with inference output and visualization
        cv::resize(original_images[i], original_images[i], cv::Size(640, 640));
    }

    // Class labels for display (customize based on your model's classes)
    std::map<int, std::string> common_labels = {
        {0, "person"},
        {1, "bicycle"},
        {2, "car"},
        // Add more labels as needed for other classes your model might detect
    };

    // Create an instance of our C++ VisionDrawer (from vision_drawer_cpp.h/cpp)
    VisionDrawer drawer(0.5f, 0.3f); // object_conf_threshold = 0.5, point_conf_threshold = 0.3

    cv::namedWindow("Inference Results", cv::WINDOW_NORMAL); // Create a resizable window

    int current_image_idx = 0;
    while (true) {
        // Create a copy of the current original image to draw on
        // This ensures we always start with a clean image for drawing
        cv::Mat display_frame = original_images[current_image_idx].clone();

        // Draw pose detections on the current image
        // We use all_results[current_image_idx] which contains the poses for this specific image
        drawer.drawPose(display_frame, all_results[current_image_idx], common_labels);

        // Display the image
        std::string title = "Inference Results - " + image_names[current_image_idx] +
                            " (" + std::to_string(current_image_idx + 1) + "/" +
                            std::to_string(original_images.size()) + ")";
        cv::imshow("Inference Results", display_frame);
        cv::setWindowTitle("Inference Results", title); // Update window title

        // Wait for a key press
        int key = cv::waitKey(0); // Wait indefinitely for a key press

        // Handle key presses
        if (key == 27) { // ESC key
            break;
        } else if (key == 81) { // Left arrow key (ASCII for left arrow on some systems)
            current_image_idx = (current_image_idx - 1 + original_images.size()) % original_images.size();
        } else if (key == 83) { // Right arrow key (ASCII for right arrow on some systems)
            current_image_idx = (current_image_idx + 1) % original_images.size();
        }
        // Note: Key codes can vary across OS/environments. 81 and 83 are common for arrow keys.
        // You might need to use a debugger or print 'key' to find exact codes if they don't work.
        // For example: std::cout << "Key pressed: " << key << std::endl;
    }

    cv::destroyAllWindows(); // Close all OpenCV windows

    return 0;
}