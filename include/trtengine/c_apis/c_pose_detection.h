#ifndef C_POSE_DETECTION_H
#define C_POSE_DETECTION_H

#include "trtengine/c_apis/c_dstruct.h"

// 对于目前的Jetson平台来说
// YoloV8的模型，一次性推理8张图片最合适的，EfficientNet的模型，一次性推理32张图片是合适的
// 因此，对于目前采用硬编码，指定了YoloPose最多一次性处理8张图片，而EfficientNet为32张图片
// 以后如果还有时间，这个参数的配置情况将全部挪到配置文件中

#ifdef __cplusplus
extern "C" {
#endif
    
    /**
     * @brief 初始化姿态检测引擎
     *
     * @param yolo_engine_path yolo pose引擎路径 (C字符串)
     * @param efficient_engine_path efficientnet引擎路径 (C字符串)
     * @param max_items 最大检测物体数量 (per batch)
     * @param cls 置信度阈值
     * @param iou IOU阈值
     * @return bool 是否成功初始化 (1 for true, 0 for false)
     */
    bool init_pose_detection_pipeline(const char* yolo_engine_path, const char* efficient_engine_path,
                                      int max_items, float cls, float iou);

    /**
     * @brief 将一张图片添加到姿态检测管道中。
     * 这个函数会将图片数据添加到内部的处理队列中，供后续处理。
     * 注意：此函数不会立即执行检测，而是将图片数据存储起来，
     * 供 run_pose_detection_pipeline 函数调用时使用。
     * @param image_data_in_bgr 输入的BGR格式图片数据指针
     * @param width 图片宽度
     * @param height 图片高度
     */
    void add_image_to_pose_detection_pipeline(const unsigned char* image_data_in_bgr, int width, int height);

    /**
     * @brief 对排队的所有图片进行姿态检测和信息扩充。
     * 这是一个阻塞调用，内部会执行预处理、推理和后处理。
     * 调用成功后，内部队列 `g_image_queue` 将被清空。
     *
     * @param out_results 指向 C_InferenceResult 数组的指针的指针。
     * 函数会内部为此数组及其嵌套的 `detections` 分配内存。
     * 调用者必须通过 `release_inference_result` 释放此内存。
     * @param out_num_results 指向整数的指针，用于存储检测结果数组的大小。
     * 这个数量将等于 `add_image_to_pose_detection_pipeline` 添加的图片数量。
     * @return bool 是否成功执行整个流程 (1 for true, 0 for false)。
     * 即使没有检测到物体，如果流程没有失败，也会返回 `true`。
     */
    bool run_pose_detection_pipeline(C_InferenceResult** out_results, int *out_num_results); // Corrected void* to C_InferenceResult**

    /**
     * @brief 销毁所有已加载的模型。
     */
    void deinit_pose_detection_pipeline();

    /**
     * @brief 释放由 `run_pose_detection_pipeline` 函数分配的 C_InferenceResult 数组及其内部的检测结果内存。
     * 注意：调用此函数后，`result_array` 指向的内存将被释放，不能再访问。
     *
     * @param result_array 指向 C_InferenceResult 数组的指针。
     * @param count 数组中的元素数量，应与 `run_pose_detection_pipeline` 返回的 `out_num_results` 相同。
     */
    void release_inference_result(C_InferenceResult* result_array, int count); // Corrected void* to C_InferenceResult* and added count

#ifdef __cplusplus
}
#endif

#endif // C_POSE_DETECTION_H