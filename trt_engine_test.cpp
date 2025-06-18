//
// Created by user on 6/13/25.
//

#include <opencv2/opencv.hpp>
#include "serverlet/models/common/yolo_drawer.h"
#include "serverlet/models/inference/model_init_helper.hpp"

// 示例用法
#include <iostream>



void test_yolo_pose()
{
    // 定义模型参数
    std::map<std::string, std::any> params;
    int maximum_batch = 4; // 明确批处理大小
    params["maximum_batch"] = maximum_batch;
    params["maximum_items"] = 100;
    params["infer_features"] = 56;
    params["infer_samples"] = 8400;

    // 用于postprocess的参数
    params["cls"] = 0.4f;
    params["iou"] = 0.5f;

    // 创建 YOLOv8 姿态估计模型
    std::unique_ptr<InferModelBaseMulti> pose_model = ModelFactory::createModel(
        "YoloV8_Pose", "/opt/models/yolov8s-pose.engine", params
    );

    // 检查模型是否成功创建
    if (pose_model) {
        std::cout << "YOLOv8 Pose Estimation Model created successfully." << std::endl;
    } else {
        std::cerr << "Failed to create YOLOv8 Pose Estimation Model." << std::endl;
        return;
    }

    // --- 加载不同的图片 ---
    std::vector<std::string> image_paths = {
        "/opt/images/human_and_pets.png",
        "/opt/images/apples.png",
        "/opt/images/cartoon.png",
        "/opt/images/pedestrian.png"
    };

    std::vector<cv::Mat> original_images(maximum_batch);
    for (int i = 0; i < maximum_batch; ++i) {
        if (i < image_paths.size()) {
            original_images[i] = cv::imread(image_paths[i]);
            if (original_images[i].empty()) {
                std::cerr << "Failed to load image: " << image_paths[i] << std::endl;
                // 如果图片加载失败，可以考虑跳过或用一个占位符图片
                // 这里我们直接返回，实际应用中可能需要更健壮的错误处理
                return;
            }
            std::cout << "Loaded image: " << image_paths[i] << std::endl;
        } else {
            // 如果图片数量少于 maximum_batch，可以重复使用最后一张图片或跳过
            std::cerr << "Warning: Not enough unique images for batch size " << maximum_batch << ". Reusing images." << std::endl;
            original_images[i] = original_images.back(); // 重复最后一张
        }
    }

    // --- 预处理所有图片，并将其拷贝到对应的批次索引 ---
    for (int i = 0; i < maximum_batch; ++i) {
        pose_model->preprocess(original_images[i], i);
    }

    // --- 执行推理 ---
    pose_model->inference();

    // --- 后处理和显示每张图片的结果 ---
    YoloDrawer drawer; // 只需要一个绘图器实例

    for (int i = 0; i < maximum_batch; ++i) {
        std::any raw_results;
        // 对每个批次索引调用 postprocess
        pose_model->postprocess(i, params, raw_results);

        try {
            auto current_batch_results = std::any_cast<std::vector<YoloPose>>(raw_results);

            // 获取当前批次的原始图片进行绘制
            cv::Mat display_image = original_images[i].clone(); // 克隆一份，避免修改原始图片
            
            // 对图片resize到模型期望的输入大小
            cv::resize(display_image, display_image, cv::Size(640, 640));

            // 绘制当前批次的姿态估计结果
            drawer.drawPoses(display_image, current_batch_results);

            // 显示结果
            std::string window_name = "Pose Detection Result - Image " + std::to_string(i + 1);
            cv::imshow(window_name, display_image);

        } catch (const std::bad_any_cast& e) {
            std::cerr << "Error: Failed to cast postprocess result for batch " << i << ": " << e.what() << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "An unexpected error occurred during YOLOv8 postprocessing for batch " << i << ": " << e.what() << std::endl;
        }
    }

    cv::waitKey(0); // 等待按键事件，直到所有窗口被关闭
    cv::destroyAllWindows();
    std::cout << "Pose detection completed and results displayed for all images." << std::endl;
}


void test_yolo_detection() 
{
    // 定义模型参数
    std::map<std::string, std::any> params;
    params["maximum_batch"] = 1;
    params["maximum_items"] = 100;
    params["infer_features"] = 84;
    params["infer_samples"] = 8400;

    // 用于postprocess的参数
    params["cls"] = 0.4f;
    params["iou"] = 0.5f;

    // 创建 YOLOv8 姿态估计模型
    std::unique_ptr<InferModelBaseMulti> detection_model = ModelFactory::createModel(
        "YoloV8_Detection", "/opt/models/yolov8s.engine", params
    );

    // 检查模型是否成功创建
    if (detection_model) {
        std::cout << "YOLOv8 Detection Model created successfully." << std::endl;
    } else {
        std::cerr << "Failed to create YOLOv8 Detection Model." << std::endl;
        return;
    }

    // --- 加载图片 ---
    std::string image_path = "/opt/images/india_road.png";
    cv::Mat original_image = cv::imread(image_path);
    if (original_image.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return; // 如果图片加载失败，可以考虑跳过或用一个占位符图片
    }

    // --- 预处理图片，并将其拷贝到对应的批次索引 ---
    detection_model->preprocess(original_image, 0);
    std::cout << "Image preprocessed and copied to batch index 0." << std::endl;

    // --- 执行推理 ---
    if (detection_model->inference()) {
        std::cout << "Inference executed successfully." << std::endl;
    } else {
        std::cerr << "Inference failed." << std::endl;
        return;
    }

    // --- 后处理和显示结果 ---
    YoloDrawer drawer; // 只需要一个绘图器实例
    std::any raw_results;
    detection_model->postprocess(0, params, raw_results);

    try {
        auto detection_results = std::any_cast<std::vector<Yolo>>(raw_results);

        // 在这里处理检测结果，例如绘制边界框
        cv::Mat display_image = original_image.clone(); // 克隆一份，避免修改原始图片
        cv::resize(display_image, display_image, cv::Size(640, 640)); // 调整大小到模型期望的输入大小

        // 绘制检测结果
        drawer.drawBoundingBoxes(display_image, detection_results);

        // 显示结果
        cv::imshow("YOLOv8 Detection Result", display_image);
        cv::waitKey(0);

    } catch (const std::bad_any_cast& e) {
        std::cerr << "Error: Failed to cast postprocess result: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "An unexpected error occurred during YOLOv8 postprocessing: " << e.what() << std::endl;
    }

    cv::destroyAllWindows();
    std::cout << "Detection completed and results displayed." << std::endl;

    // 执行超 maximum_batch 测试
    detection_model->preprocess(original_image, 1); // 尝试使用批次索引 1
    detection_model->preprocess(original_image, 2); // 尝试使用批次索引 2
    detection_model->preprocess(original_image, 3); // 尝试使用批次索引 3
    detection_model->preprocess(original_image, 4); // 尝试使用批次索引 4
}


void test_yolo_pose_efficient()
{
    // 定义YoloPose模型参数
    std::map<std::string, std::any> params1;
    params1["maximum_batch"] = 1;
    params1["maximum_items"] = 100;
    params1["infer_features"] = 56;
    params1["infer_samples"] = 8400;

    // 用于postprocess的参数
    params1["cls"] = 0.4f;
    params1["iou"] = 0.5f;

    // 创建 YOLOv8 姿态估计模型
    std::unique_ptr<InferModelBaseMulti> pose_model = ModelFactory::createModel(
        "YoloV8_Pose", "/opt/models/yolov8s-pose.engine", params1
    );
    if (!pose_model) {
        std::cerr << "Failed to create YOLOv8 Pose Estimation Model." << std::endl;
        return;
    }

    // 定义EfficientNet模型参数
    std::map<std::string, std::any> params2;
    params2["maximum_batch"] = 4;

    // 创建 EfficientNet 模型
    std::unique_ptr<InferModelBaseMulti> efficient_model = ModelFactory::createModel(
        "EfficientNet", "/opt/models/efficientnet_b0_feat_logits.engine", params2
    );
    if (!efficient_model) {
        std::cerr << "Failed to create EfficientNet Model." << std::endl;
        return;
    }

    // --- 加载图片 ---
    std::string image_path = "/opt/images/supermarket/customer4.png";
    cv::Mat original_image = cv::imread(image_path);
    if (original_image.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return; // 如果图片加载失败，可以考虑跳过或用一个占位符图片
    }

    // 首先预处理 YOLOv8 姿态估计模型
    pose_model->preprocess(original_image, 0);
    std::cout << "Pose model image preprocessed and copied to batch index 0." << std::endl;
    // 执行 YOLOv8 姿态估计模型推理
    if (pose_model->inference()) {
        std::cout << "Pose model inference executed successfully." << std::endl;
    } else {
        std::cerr << "Pose model inference failed." << std::endl;
        return;
    }

    // 后处理 YOLOv8 姿态估计模型
    std::any pose_results;
    pose_model->postprocess(0, params1, pose_results);
    std::vector<YoloPose> pose_detections;
    try {
        pose_detections = std::any_cast<std::vector<YoloPose>>(pose_results);
    } catch (const std::bad_any_cast& e) {
        std::cerr << "Error casting pose results: " << e.what() << std::endl;
        return;
    }

    // 一共检测到多少个人体
    std::cout << "Detected " << pose_detections.size() << " persons in the image." << std::endl;

    // 将图片resize到模型期望的输入大小
    cv::Mat resized_image;
    cv::resize(original_image, resized_image, cv::Size(640, 640));

    // 根据检测到的人体数量，从图片中裁剪出对应的区域
    const float scale_factor = 1.2f; // 缩放因子
    std::vector<cv::Mat> cropped_images;
    for (const auto& pose : pose_detections) {
        if (pose.pts.empty()) {
            std::cerr << "No keypoints detected for a person, skipping." << std::endl;
            continue;
        }

        // 计算人体边界框
        int min_x = std::min(pose.lx, pose.rx);
        int min_y = std::min(pose.ly, pose.ry);
        int max_x = std::max(pose.lx, pose.rx);
        int max_y = std::max(pose.ly, pose.ry);
        int width = max_x - min_x;
        int height = max_y - min_y;

        // 计算裁剪区域，添加缩放因子
        int crop_x = std::max(0, static_cast<int>(min_x - width * (scale_factor - 1) / 2));
        int crop_y = std::max(0, static_cast<int>(min_y - height * (scale_factor - 1) / 2));
        int crop_width = std::min(resized_image.cols - crop_x, static_cast<int>(width * scale_factor));
        int crop_height = std::min(resized_image.rows - crop_y, static_cast<int>(height * scale_factor));
        cv::Mat cropped_image = resized_image(cv::Rect(crop_x, crop_y, crop_width, crop_height));
        cropped_images.push_back(cropped_image);
    }

    // 现在批量处理裁剪后的图片
    for (size_t i = 0; i < cropped_images.size(); ++i)
    {
        // 预处理 EfficientNet 模型
        efficient_model->preprocess(cropped_images[i], i);
        std::cout << "EfficientNet model image preprocessed and copied to batch index " << i << "." << std::endl;
    }

    // 执行 EfficientNet 模型推理
    if (efficient_model->inference()) {
        std::cout << "EfficientNet model inference executed successfully." << std::endl;
    } else {
        std::cerr << "EfficientNet model inference failed." << std::endl;
        return;
    }

    // --- 后处理 EfficientNet 模型，遍历所有裁剪图片对应的批次 ---
    for (size_t i = 0; i < cropped_images.size(); ++i) { // 遍历所有有效批次
        std::any efficient_results_per_person;
        efficient_model->postprocess(i, params2, efficient_results_per_person);

        try {
            auto current_person_feat_cls = std::any_cast<std::vector<float>>(efficient_results_per_person);

            if (!current_person_feat_cls.empty()) {
                int predicted_class = static_cast<int>(current_person_feat_cls[0]);
                std::cout << "  Person " << i << ": Predicted class: " << predicted_class
                          << ", Feature vector size: " << current_person_feat_cls.size() - 1 << std::endl;

                // **核心修改：使用引用来更新原始向量中的元素**
                auto& pose = pose_detections[i]; // 注意这里的 '&'
                pose.cls = static_cast<float>(predicted_class); // 确保 cls 是 float 类型

            } else {
                std::cout << "  Person " << i << ": No EfficientNet results found." << std::endl;
            }

        } catch (const std::bad_any_cast& e) {
            std::cerr << "Error casting EfficientNet results for person " << i << ": " << e.what() << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "An unexpected error occurred during EfficientNet postprocessing for person " << i << ": " << e.what() << std::endl;
        }
    }
    std::cout << "EfficientNet postprocessing completed for all detected persons." << std::endl;

    // Optional: Draw poses on the original image (if you want to visualize both)
    YoloDrawer drawer;
    cv::Mat display_image_combined = original_image.clone();
    cv::resize(display_image_combined, display_image_combined, cv::Size(640, 640)); // Resize for consistency
    drawer.drawPoses(display_image_combined, pose_detections);

    cv::imshow("Combined Results (YOLOv8 Pose + EfficientNet)", display_image_combined);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

void unknown_model_test() {
    // 尝试创建不存在的模型
    std::unique_ptr<InferModelBaseMulti> unknown_model = ModelFactory::createModel(
        "UnknownModel", "path/to/unknown_engine.trt", {}
    );

    if (!unknown_model) {
        std::cout << "Correctly failed to create UnknownModel." << std::endl;
    } else {
        std::cout << "Unexpectedly created UnknownModel." << std::endl;
    }
}

int main() {

    // 注册所有模型
    registerModels();

    // 黄色输出前缀和后缀
    const std::string YELLOW = "\033[33m";
    const std::string RESET = "\033[0m";
    const std::string YELLOW_LINE = YELLOW + "-------------------" + RESET;

    // 测试 YOLOv8 姿态估计模型
    std::cout << YELLOW << "Detecting poses using YOLOv8..." << RESET << std::endl;
    test_yolo_pose();
    std::cout << YELLOW_LINE << std::endl;

    // 测试 YOLOv8 检测模型
    std::cout << YELLOW << "Detecting objects using YOLOv8..." << RESET << std::endl;
    test_yolo_detection();
    std::cout << YELLOW_LINE << std::endl;

    // 测试 YOLOv8 + EfficientNet 模型
    std::cout << YELLOW << "Testing YOLOv8 + EfficientNet..." << RESET << std::endl;
    test_yolo_pose_efficient();
    std::cout << YELLOW_LINE << std::endl;

    // 尝试创建不存在的模型
    std::cout << YELLOW << "Testing unknown model creation..." << RESET << std::endl;
    unknown_model_test();
    std::cout << YELLOW_LINE << std::endl;

    return 0;
}