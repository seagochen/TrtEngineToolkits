//
// Created by user on 6/17/25.
//

#ifndef VISION_DRAWER_H
#define VISION_DRAWER_H

#include <opencv2/opencv.hpp>
#include "serverlet/models/common/yolo_dstruct.h"

// Define colors in BGR format
const cv::Scalar BLUE = cv::Scalar(255, 0, 0);
const cv::Scalar GREEN = cv::Scalar(0, 255, 0);
const cv::Scalar RED = cv::Scalar(0, 0, 255);
const cv::Scalar CYAN = cv::Scalar(255, 255, 0);
const cv::Scalar MAGENTA = cv::Scalar(255, 0, 255);
const cv::Scalar YELLOW = cv::Scalar(0, 255, 255);
const cv::Scalar WHITE = cv::Scalar(255, 255, 255);
const cv::Scalar BLACK = cv::Scalar(0, 0, 0);
const cv::Scalar ORANGE = cv::Scalar(0, 165, 255);
const cv::Scalar PURPLE = cv::Scalar(128, 0, 128);

// Helper struct for KeyPoint information, similar to Python's KeyPoint dataclass
struct KeyPointInfo {
    std::string name;
    cv::Scalar color;
};

// Helper struct for Skeleton information, similar to Python's Skeleton dataclass
struct SkeletonInfo {
    int srt_kpt_id;
    int dst_kpt_id;
    cv::Scalar color;
};

class VisionDrawer {

public:
    VisionDrawer(float object_conf_threshold = 0.25f, float point_conf_threshold = 0.25f);

    // Draw bounding boxes for a vector of Yolo objects
    void drawBBoxes(cv::Mat& frame, const std::vector<Yolo>& detections,
                    const std::map<int, std::string>& class_labels = {});

    // Draw bounding boxes, keypoints, and skeletons for a vector of YoloPose objects
    void drawPose(cv::Mat& frame, const std::vector<YoloPose>& pose_detections,
                  const std::map<int, std::string>& class_labels = {});

private:
    float object_conf_threshold;
    float point_conf_threshold;

    // Default color palettes and schema definitions
    std::vector<cv::Scalar> bbox_colors;
    std::map<int, KeyPointInfo> kpt_color_map;
    std::vector<SkeletonInfo> skeleton_map;

    void _load_default_schema();

    // Helper to get bounding box color based on class ID
    cv::Scalar _get_bbox_color_by_class(int class_id) const;

    // Helper to draw a single bounding box with label
    void _draw_bbox_and_label(cv::Mat& image,
                              const std::string& text,
                              const cv::Rect& bbox_rect,
                              const cv::Scalar& bbox_color,
                              const cv::Scalar& text_color = WHITE,
                              double font_scale = 0.5,
                              int thickness = 1);
};

#endif //VISION_DRAWER_H
