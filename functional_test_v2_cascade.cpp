/**
 * @file functional_test_v2_cascade.cpp
 * @brief V2 级联推理测试程序
 *
 * 功能流程:
 * 1. YOLOv8-Pose 检测人物和关键点
 * 2. 绘制关键点和骨架连线
 * 3. 裁剪检测到的人物区域
 * 4. EfficientNet 对裁剪区域进行分类和特征提取
 *
 * @author TrtEngineToolkits
 * @date 2025-11-10
 */

#include "trtengine_v2/pipelines/yolopose/c_yolopose_pipeline.h"
#include "trtengine_v2/pipelines/efficientnet/c_efficientnet_pipeline.h"
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ============================================================================
//                         COCO 骨架连接定义
// ============================================================================

// COCO 骨架连接定义 (17个关键点)
static const int SKELETON_CONNECTIONS[][2] = {
    {0, 1}, {0, 2},   // 鼻子到眼睛
    {1, 3}, {2, 4},   // 眼睛到耳朵
    {0, 5}, {0, 6},   // 鼻子到肩膀
    {5, 7}, {7, 9},   // 左臂
    {6, 8}, {8, 10},  // 右臂
    {5, 6},           // 肩膀连接
    {5, 11}, {6, 12}, // 肩膀到髋部
    {11, 12},         // 髋部连接
    {11, 13}, {13, 15}, // 左腿
    {12, 14}, {14, 16}  // 右腿
};

static const int NUM_SKELETON_CONNECTIONS = sizeof(SKELETON_CONNECTIONS) / sizeof(SKELETON_CONNECTIONS[0]);

// ============================================================================
//                         图像处理函数
// ============================================================================

/**
 * @brief 根据关键点索引获取颜色
 */
cv::Scalar get_keypoint_color(int kpt_idx) {
    // OpenCV 使用 BGR 格式
    if (kpt_idx == 0) {
        return cv::Scalar(0, 0, 255);        // 红色 - 鼻子
    } else if (kpt_idx >= 1 && kpt_idx <= 4) {
        return cv::Scalar(255, 0, 0);        // 蓝色 - 眼睛和耳朵
    } else if (kpt_idx >= 5 && kpt_idx <= 6) {
        return cv::Scalar(0, 255, 0);        // 绿色 - 肩膀
    } else if (kpt_idx >= 7 && kpt_idx <= 10) {
        return cv::Scalar(255, 255, 0);      // 青色 - 手臂
    } else {
        return cv::Scalar(255, 0, 255);      // 品红色 - 腿部
    }
}

/**
 * @brief 在图像上绘制单个姿态 (边界框 + 关键点 + 骨架)
 */
void draw_pose(cv::Mat& image, const C_YoloPose* pose, float conf_threshold = 0.3f) {
    // 1. 绘制边界框
    cv::rectangle(image,
                  cv::Point(pose->detection.lx, pose->detection.ly),
                  cv::Point(pose->detection.rx, pose->detection.ry),
                  cv::Scalar(0, 255, 0),  // 绿色
                  2);

    // 2. 绘制骨架连线
    for (int i = 0; i < NUM_SKELETON_CONNECTIONS; i++) {
        int idx1 = SKELETON_CONNECTIONS[i][0];
        int idx2 = SKELETON_CONNECTIONS[i][1];

        const C_KeyPoint* p1 = &pose->pts[idx1];
        const C_KeyPoint* p2 = &pose->pts[idx2];

        // 只绘制置信度足够高的连线
        if (p1->conf > conf_threshold && p2->conf > conf_threshold) {
            cv::line(image,
                     cv::Point((int)p1->x, (int)p1->y),
                     cv::Point((int)p2->x, (int)p2->y),
                     cv::Scalar(0, 255, 255),  // 黄色 (BGR)
                     2);
        }
    }

    // 3. 绘制关键点 (在连线之上)
    for (int i = 0; i < 17; i++) {
        const C_KeyPoint* kpt = &pose->pts[i];
        if (kpt->conf > conf_threshold) {
            cv::Scalar color = get_keypoint_color(i);
            cv::circle(image,
                       cv::Point((int)kpt->x, (int)kpt->y),
                       5,
                       color,
                       -1);  // 填充圆
        }
    }
}

// ============================================================================
//                         主程序
// ============================================================================

int main(int argc, char** argv) {
    if (argc != 4) {
        printf("Usage: %s <yolopose_engine> <efficientnet_engine> <input_image>\n", argv[0]);
        printf("\nExample:\n");
        printf("  %s yolov8n-pose.engine efficientnet_b0.engine people.jpg\n", argv[0]);
        return 1;
    }

    const char* yolopose_engine_path = argv[1];
    const char* efficientnet_engine_path = argv[2];
    const char* input_image_path = argv[3];

    printf("========================================\n");
    printf("V2 级联推理测试程序\n");
    printf("========================================\n");
    printf("YOLOv8-Pose Engine: %s\n", yolopose_engine_path);
    printf("EfficientNet Engine: %s\n", efficientnet_engine_path);
    printf("Input Image: %s\n", input_image_path);
    printf("========================================\n\n");

    // ========================================================================
    // [1/6] 加载输入图像
    // ========================================================================
    printf("[1/6] 加载输入图像...\n");
    cv::Mat input_image = cv::imread(input_image_path);
    if (input_image.empty()) {
        printf("  错误: 无法加载图像 '%s'\n", input_image_path);
        return 1;
    }

    // 转换为 RGB (OpenCV 默认是 BGR)
    cv::Mat input_image_rgb;
    cv::cvtColor(input_image, input_image_rgb, cv::COLOR_BGR2RGB);

    printf("  图像尺寸: %dx%d, 通道数: %d\n",
           input_image_rgb.cols, input_image_rgb.rows, input_image_rgb.channels());

    // ========================================================================
    // [2/6] 创建 YOLOv8-Pose Pipeline
    // ========================================================================
    printf("\n[2/6] 创建 YOLOv8-Pose Pipeline...\n");
    C_YoloPosePipelineConfig pose_config = c_yolopose_pipeline_get_default_config();
    pose_config.engine_path = yolopose_engine_path;
    pose_config.conf_threshold = 0.25f;
    pose_config.iou_threshold = 0.45f;

    C_YoloPosePipelineContext* pose_pipeline = c_yolopose_pipeline_create(&pose_config);
    if (!pose_pipeline) {
        printf("  错误: 无法创建 YOLOv8-Pose Pipeline\n");
        return 1;
    }
    printf("  Pipeline 创建成功\n");

    // ========================================================================
    // [3/6] 运行姿态检测
    // ========================================================================
    printf("\n[3/6] 运行姿态检测...\n");

    C_ImageInput image_input;
    image_input.data = input_image_rgb.data;
    image_input.width = input_image_rgb.cols;
    image_input.height = input_image_rgb.rows;
    image_input.channels = input_image_rgb.channels();

    C_YoloPoseImageResult pose_result = {0};
    if (!c_yolopose_infer_single(pose_pipeline, &image_input, &pose_result)) {
        printf("  错误: 姿态检测失败: %s\n",
               c_yolopose_pipeline_get_last_error(pose_pipeline));
        c_yolopose_pipeline_destroy(pose_pipeline);
        return 1;
    }

    printf("  检测到 %zu 个人\n", pose_result.num_poses);

    if (pose_result.num_poses == 0) {
        printf("  警告: 未检测到任何人物，程序退出\n");
        c_yolopose_image_result_free(&pose_result);
        c_yolopose_pipeline_destroy(pose_pipeline);
        return 0;
    }

    // 输出检测结果
    for (size_t i = 0; i < pose_result.num_poses; i++) {
        const C_YoloPose* pose = &pose_result.poses[i];
        printf("  Person %zu: bbox=[%d,%d,%d,%d], conf=%.2f\n",
               i, pose->detection.lx, pose->detection.ly,
               pose->detection.rx, pose->detection.ry,
               pose->detection.conf);
    }

    // 绘制姿态到图像上 (BGR 格式)
    cv::Mat pose_image = input_image.clone();
    for (size_t i = 0; i < pose_result.num_poses; i++) {
        draw_pose(pose_image, &pose_result.poses[i]);
    }

    // 保存姿态图像
    const char* pose_output_path = "output_pose.jpg";
    cv::imwrite(pose_output_path, pose_image);
    printf("  已保存姿态图像: %s\n", pose_output_path);

    // ========================================================================
    // [4/6] 创建 EfficientNet Pipeline
    // ========================================================================
    printf("\n[4/6] 创建 EfficientNet Pipeline...\n");
    C_EfficientNetPipelineConfig eff_config = c_efficientnet_pipeline_get_default_config();
    eff_config.engine_path = efficientnet_engine_path;

    C_EfficientNetPipelineContext* eff_pipeline = c_efficientnet_pipeline_create(&eff_config);
    if (!eff_pipeline) {
        printf("  错误: 无法创建 EfficientNet Pipeline\n");
        c_yolopose_image_result_free(&pose_result);
        c_yolopose_pipeline_destroy(pose_pipeline);
        return 1;
    }
    printf("  Pipeline 创建成功\n");

    // ========================================================================
    // [5/6] 裁剪人物区域并进行分类
    // ========================================================================
    printf("\n[5/6] 裁剪人物区域并进行分类...\n");

    for (size_t i = 0; i < pose_result.num_poses; i++) {
        const C_YoloPose* pose = &pose_result.poses[i];
        printf("\n--- Person %zu ---\n", i);

        // 裁剪边界框区域
        int crop_lx = std::max(0, pose->detection.lx);
        int crop_ly = std::max(0, pose->detection.ly);
        int crop_rx = std::min(input_image_rgb.cols, pose->detection.rx);
        int crop_ry = std::min(input_image_rgb.rows, pose->detection.ry);

        cv::Rect crop_rect(crop_lx, crop_ly, crop_rx - crop_lx, crop_ry - crop_ly);
        cv::Mat cropped_rgb = input_image_rgb(crop_rect);

        printf("  裁剪区域: %dx%d\n", cropped_rgb.cols, cropped_rgb.rows);

        // 保存裁剪图像 (BGR 格式)
        cv::Mat cropped_bgr;
        cv::cvtColor(cropped_rgb, cropped_bgr, cv::COLOR_RGB2BGR);
        char crop_filename[256];
        snprintf(crop_filename, sizeof(crop_filename), "output_crop_%zu.jpg", i);
        cv::imwrite(crop_filename, cropped_bgr);
        printf("  已保存裁剪图像: %s\n", crop_filename);

        // EfficientNet 推理
        C_ImageInput crop_input;
        crop_input.data = cropped_rgb.data;
        crop_input.width = cropped_rgb.cols;
        crop_input.height = cropped_rgb.rows;
        crop_input.channels = cropped_rgb.channels();

        C_EfficientNetResult eff_result = {0};
        if (!c_efficientnet_infer_single(eff_pipeline, &crop_input, &eff_result)) {
            printf("  错误: EfficientNet 推理失败: %s\n",
                   c_efficientnet_pipeline_get_last_error(eff_pipeline));
            continue;
        }

        // 输出分类结果
        printf("  分类结果:\n");
        printf("    预测类别: %d\n", eff_result.class_id);
        printf("    置信度: %.4f\n", eff_result.confidence);
        printf("    Logits: ");
        for (size_t j = 0; j < eff_result.num_classes; j++) {
            printf("%.4f ", eff_result.logits[j]);
        }
        printf("\n");

        // 输出特征向量信息
        printf("  特征向量信息:\n");
        printf("    维度: %zu\n", eff_result.feature_size);

        // 计算 L2 范数
        float l2_norm = 0.0f;
        for (size_t j = 0; j < eff_result.feature_size; j++) {
            l2_norm += eff_result.features[j] * eff_result.features[j];
        }
        l2_norm = sqrtf(l2_norm);
        printf("    L2 范数: %.4f\n", l2_norm);

        // 输出前 10 个特征值
        printf("    前 10 个特征值: ");
        for (size_t j = 0; j < 10 && j < eff_result.feature_size; j++) {
            printf("%.4f ", eff_result.features[j]);
        }
        printf("...\n");

        // 清理结果
        c_efficientnet_result_free(&eff_result);
    }

    // ========================================================================
    // [6/6] 清理资源
    // ========================================================================
    printf("\n[6/6] 清理资源...\n");
    c_yolopose_image_result_free(&pose_result);
    c_yolopose_pipeline_destroy(pose_pipeline);
    c_efficientnet_pipeline_destroy(eff_pipeline);
    printf("  完成\n");

    printf("\n========================================\n");
    printf("处理完成！\n");
    printf("========================================\n");
    printf("输出文件:\n");
    printf("  - output_pose.jpg       : 带关键点和骨架的图像\n");
    printf("  - output_crop_*.jpg     : 裁剪的人物区域\n");
    printf("========================================\n");

    return 0;
}
