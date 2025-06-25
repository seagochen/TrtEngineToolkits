//
// Created by user on 6/25/25.
//

#include <opencv2/opencv.hpp>
#include "trtengine/serverlet/models/inference/model_init_helper.hpp"
#include "trtengine/utils/system.h" // Assuming getCurrentRSS is in this header

// Example usage
#include <iostream>
#include <vector>
#include <string>
#include <algorithm> // For std::min, std::max
#include <any>       // For std::any
#include <map>       // For std::map
#include <chrono>    // For std::chrono
#include <numeric>   // For std::accumulate (if calculating average throughput)
#include <thread>         // For std::thread
#include <queue>          // For std::queue
#include <mutex>          // For std::mutex
#include <condition_variable> // For std::condition_variable
#include <atomic>         // For std::atomic_bool

// Helper function to draw pose detection results
void draw_pose_results(cv::Mat& image, const std::vector<YoloPose>& pose_detections) {
    for (const auto& pose : pose_detections) {
        // Choose color based on pose.cls
        cv::Scalar box_color;
        // Assuming cls is a float between 0 and 1
        // We can segment based on cls value to choose colors
        if (pose.cls < 0.2f) {
            box_color = cv::Scalar(0, 0, 255); // Red for low confidence
        } else if (pose.cls < 0.5f) {
            box_color = cv::Scalar(0, 165, 255); // Orange for medium-low confidence
        } else if (pose.cls < 0.8f) {
            box_color = cv::Scalar(0, 255, 255); // Yellow for medium-high confidence
        } else {
            box_color = cv::Scalar(0, 255, 0); // Green for high confidence
        }

        // Draw bounding box
        cv::rectangle(image, cv::Rect(pose.lx, pose.ly, pose.rx - pose.lx, pose.ry - pose.ly), box_color, 2);

        // Draw keypoints (keypoint color can usually be fixed, or also changed based on cls)
        // We'll keep keypoints red here for distinction
        for (const auto& pt : pose.pts) {
            if (pt.x >= 0 && pt.y >= 0) { // Ensure keypoint is valid
                cv::circle(image, cv::Point(pt.x, pt.y), 3, cv::Scalar(0, 0, 255), -1);
            }
        }
        // Draw class score (optional)
        std::string label = "Cls: " + std::to_string(pose.cls);
        cv::putText(image, label, cv::Point(pose.lx, pose.ly - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1); // Text color matches box
    }
}

// Data structure to pass between threads
struct PoseResultWithID {
    int iteration_id;
    std::vector<YoloPose> detections;
    bool is_final_signal; // To signal termination (poison pill)

    // Constructor for normal results
    PoseResultWithID(int id, const std::vector<YoloPose>& det)
        : iteration_id(id), detections(det), is_final_signal(false) {}

    // Constructor for final signal (poison pill)
    PoseResultWithID() : iteration_id(-1), is_final_signal(true) {}
};

// Thread-safe queue for YoloPose results
std::queue<PoseResultWithID> pose_results_queue;
std::mutex queue_mutex;
std::condition_variable queue_cv;
std::atomic<bool> yolo_producer_done(false); // Flag to indicate YoloPose thread has finished producing

// Global vectors to store durations from each thread
// Note: In a larger application, these might be part of a shared context
// or passed as references to avoid global state. For a benchmark, it's acceptable.
std::vector<long long> yolo_preprocess_times_thread;
std::vector<long long> yolo_inference_times_thread;
std::vector<long long> yolo_postprocess_times_thread;

std::vector<long long> crop_times_thread;
std::vector<long long> efficient_preprocess_times_thread;
std::vector<long long> efficient_inference_times_thread;
std::vector<long long> efficient_postprocess_times_thread;


// YoloPose thread function (Producer)
void yolo_pose_thread_func(int num_iterations, const cv::Mat& resized_image,
                           const std::map<std::string, std::any>& params1) {

    // Initialize model within the thread
    std::unique_ptr<InferModelBaseMulti> pose_model = ModelFactory::createModel("YoloV8_Pose",
        "/opt/models/yolov8n-pose.engine", params1);
    if (!pose_model) {
        std::cerr << "YoloPose Thread: Failed to create pose model. Exiting thread." << std::endl;
        return;
    }

    for (int iter = 0; iter < num_iterations; ++iter) {
        // YoloPose Preprocess
        auto step_start_time = std::chrono::high_resolution_clock::now();
        pose_model->preprocess(resized_image, 0); // Using stream 0 (default)
        yolo_preprocess_times_thread.push_back(
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - step_start_time).count()
        );

        // YoloPose Inference
        step_start_time = std::chrono::high_resolution_clock::now();
        pose_model->inference();
        yolo_inference_times_thread.push_back(
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - step_start_time).count()
        );

        // YoloPose Postprocess
        step_start_time = std::chrono::high_resolution_clock::now();
        std::any pose_results;
        pose_model->postprocess(0, params1, pose_results);
        yolo_postprocess_times_thread.push_back(
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - step_start_time).count()
        );

        std::vector<YoloPose> current_pose_detections;
        try {
            current_pose_detections = std::any_cast<std::vector<YoloPose>>(pose_results);
        } catch (...) {
            // If cast fails, send an empty detection set
            current_pose_detections.clear();
        }

        // Push results to the queue
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            pose_results_queue.push(PoseResultWithID(iter, current_pose_detections));
        }
        queue_cv.notify_one(); // Notify consumer
    }

    // Send final signal (poison pill)
    {
        std::lock_guard<std::mutex> lock(queue_mutex);
        pose_results_queue.push(PoseResultWithID()); // Send final signal
        yolo_producer_done = true; // Set atomic flag
    }
    queue_cv.notify_one(); // Notify consumer one last time
    std::cout << "YoloPose Thread: Finished producing " << num_iterations << " iterations." << std::endl;
}

// EfficientNet thread function (Consumer)
void efficient_net_thread_func(const cv::Mat& resized_image,
                               const std::map<std::string, std::any>& params2,
                               std::vector<YoloPose>& last_pose_detections_ref) { // Reference to store final detections

    // Initialize model within the thread
    std::unique_ptr<InferModelBaseMulti> efficient_model = ModelFactory::createModel("EfficientNet",
        "/opt/models/efficientnet_b0_feat_logits.engine", params2);
    if (!efficient_model) {
        std::cerr << "EfficientNet Thread: Failed to create efficient model. Exiting thread." << std::endl;
        return;
    }

    int processed_count = 0;
    while (true) {
        std::unique_lock<std::mutex> lock(queue_mutex);
        // Wait until queue is not empty OR producer is done (and queue is empty)
        queue_cv.wait(lock, [&]{ return !pose_results_queue.empty() || yolo_producer_done; });

        // Check for termination condition before popping
        if (pose_results_queue.empty() && yolo_producer_done) {
            // Producer is done and queue is empty, so consumer is done
            std::cout << "EfficientNet Thread: All items processed. Exiting." << std::endl;
            break;
        }

        PoseResultWithID frame_data = pose_results_queue.front();
        pose_results_queue.pop();
        lock.unlock(); // Release lock as soon as possible, allowing producer to push

        if (frame_data.is_final_signal) {
            std::cout << "EfficientNet Thread: Received final signal. Exiting." << std::endl;
            break; // Exit loop if poison pill received
        }

        std::vector<YoloPose>& current_pose_detections = frame_data.detections;

        // Image Cropping
        auto step_start_time = std::chrono::high_resolution_clock::now();
        const float scale_factor = 1.2f;
        std::vector<cv::Mat> cropped_images;
        size_t max_efficient_batch = 4; // From params2["maximum_batch"]
        for (size_t i = 0; i < current_pose_detections.size() && cropped_images.size() < max_efficient_batch; ++i) {
            const auto& pose = current_pose_detections[i];
            if (pose.pts.empty()) continue;

            int min_x = std::min(pose.lx, pose.rx);
            int min_y = std::min(pose.ly, pose.ry);
            int max_x = std::max(pose.lx, pose.rx);
            int max_y = std::max(pose.ly, pose.ry);
            int width = max_x - min_x;
            int height = max_y - min_y;

            int crop_x = std::max(0, static_cast<int>(min_x - width * (scale_factor - 1) / 2));
            int crop_y = std::max(0, static_cast<int>(min_y - height * (scale_factor - 1) / 2));
            int crop_width = std::min(resized_image.cols - crop_x, static_cast<int>(width * scale_factor));
            int crop_height = std::min(resized_image.rows - crop_y, static_cast<int>(height * scale_factor));

            if (crop_width > 0 && crop_height > 0)
                cropped_images.emplace_back(resized_image(cv::Rect(crop_x, crop_y, crop_width, crop_height)));
        }
        crop_times_thread.push_back(
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - step_start_time).count()
        );

        // EfficientNet Preprocess
        step_start_time = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < cropped_images.size(); ++i) {
            // Note: If EfficientNet could run multiple inferences concurrently on different CUDA streams
            // this loop would become more complex, potentially involving multiple `efficient_model` instances
            // or explicit stream management. For now, it processes cropped_images sequentially per batch.
            efficient_model->preprocess(cropped_images[i], i); // Assuming stream 'i' for batch processing if supported
        }
        efficient_preprocess_times_thread.push_back(
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - step_start_time).count()
        );

        // EfficientNet Inference
        step_start_time = std::chrono::high_resolution_clock::now();
        efficient_model->inference();
        efficient_inference_times_thread.push_back(
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - step_start_time).count()
        );

        // EfficientNet Postprocess
        step_start_time = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < cropped_images.size(); ++i) {
            std::any results;
            efficient_model->postprocess(i, params2, results);
            try {
                auto cls_result = std::any_cast<std::vector<float>>(results);
                if (!cls_result.empty() && i < current_pose_detections.size())
                    current_pose_detections[i].cls = static_cast<float>(cls_result[0]);
            } catch (...) {
                // Handle error if any_cast fails (e.g., log it)
            }
        }
        efficient_postprocess_times_thread.push_back(
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - step_start_time).count()
        );

        // Update the reference to store the last processed detections for display in main thread
        // This implicitly assumes the last frame processed is the one to display.
        last_pose_detections_ref = current_pose_detections;
        processed_count++;
    }
    std::cout << "EfficientNet Thread: Processed a total of " << processed_count << " items." << std::endl;
}

// Function to benchmark YOLO Pose and EfficientNet inference with threading
void benchmark_yolo_pose_efficient_threaded(int num_iterations = 1000, bool display_results = true) {
    std::string image_path = "/opt/images/supermarket/customer2.png";
    cv::Mat original_image = cv::imread(image_path);
    if (original_image.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return;
    }

    cv::Mat resized_image;
    cv::resize(original_image, resized_image, cv::Size(640, 640));

    // Parameters for models
    std::map<std::string, std::any> params1{
        {"maximum_batch", 1},
        {"maximum_items", 100},
        {"infer_features", 56},
        {"infer_samples", 8400},
        {"cls", 0.4f},
        {"iou", 0.5f}
    };

    std::map<std::string, std::any> params2{
        {"maximum_batch", 32}
    };

    std::vector<YoloPose> final_display_detections; // To store the very last detection for display

    auto start_total_time = std::chrono::high_resolution_clock::now();

    // Start YoloPose (producer) thread
    // std::cref is used to pass by const reference, avoiding copies
    std::thread yolo_thread(yolo_pose_thread_func, num_iterations, std::cref(resized_image), std::cref(params1));

    // Start EfficientNet (consumer) thread
    // std::ref is used for final_display_detections because the consumer thread will modify it
    std::thread efficient_thread(efficient_net_thread_func, std::cref(resized_image), std::cref(params2), std::ref(final_display_detections));

    // Wait for both threads to complete
    yolo_thread.join();
    efficient_thread.join();

    auto end_total_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_duration = end_total_time - start_total_time;

    // Calculate overall average time based on the number of iterations initiated by YoloPose
    double avg_total_ms = (total_duration.count() * 1000) / num_iterations;

    std::cout << "\n--- Threaded Efficient YOLO Pose Benchmark Results ---" << std::endl;
    std::cout << "  Total iterations (YoloPose produced): " << num_iterations << std::endl;
    std::cout << "  Total pipeline time: " << total_duration.count() << " seconds" << std::endl;
    std::cout << "  Average total pipeline time per YoloPose iteration: " << avg_total_ms << " ms" << std::endl;

    // Calculate and display average times for each step from collected thread vectors
    auto calculate_average = [](const std::vector<long long>& times) {
        if (times.empty()) return 0.0;
        long long sum = std::accumulate(times.begin(), times.end(), 0LL);
        return static_cast<double>(sum) / times.size();
    };

    std::cout << "\n--- Average Time Per Step (from each thread) ---" << std::endl;
    std::cout << "  YoloPose Preprocess (Avg): " << calculate_average(yolo_preprocess_times_thread) << " ms" << std::endl;
    std::cout << "  YoloPose Inference (Avg): " << calculate_average(yolo_inference_times_thread) << " ms" << std::endl;
    std::cout << "  YoloPose Postprocess (Avg): " << calculate_average(yolo_postprocess_times_thread) << " ms" << std::endl;
    std::cout << "  Image Cropping (Avg): " << calculate_average(crop_times_thread) << " ms" << std::endl;
    std::cout << "  EfficientNet Preprocess (Avg): " << calculate_average(efficient_preprocess_times_thread) << " ms" << std::endl;
    std::cout << "  EfficientNet Inference (Avg): " << calculate_average(efficient_inference_times_thread) << " ms" << std::endl;
    std::cout << "  EfficientNet Postprocess (Avg): " << calculate_average(efficient_postprocess_times_thread) << " ms" << std::endl;


    if (display_results) {
        if (!final_display_detections.empty()) {
            cv::Mat display_image = resized_image.clone();
            draw_pose_results(display_image, final_display_detections);
            cv::imshow("Threaded YOLO Pose Detection Results", display_image);
            cv::waitKey(0);
            cv::destroyAllWindows(); // Close the window when a key is pressed
        } else {
            std::cout << "No pose detections found in the last iteration to display." << std::endl;
        }
    }
}

int main() {
    registerModels();
    // Call the new threaded benchmark function
    benchmark_yolo_pose_efficient_threaded(1000, true);
    return 0;
}