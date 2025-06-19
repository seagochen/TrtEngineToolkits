//
// Created by user on 6/19/25.
//

#ifndef C_DSTRUCT_H
#define C_DSTRUCT_H

#ifdef __cplusplus
extern "C" {
#endif

    // 定义C风格的Point结构体
    typedef struct C_Point {
        float x;
        float y;
        float score; // 关键点置信度
    } C_Point;

    // 定义C风格的边界框结构体 (x1, y1, x2, y2)
    typedef struct C_Rect {
        float x1;
        float y1;
        float x2;
        float y2;
    } C_Rect;


    // 定义C风格的扩展人物特征结构体
    // 对应C++的YoloPose，但增加了分类信息
    typedef struct C_Extended_Person_Feats {
        C_Rect box;      // 边界框
        float confidence; // 检测置信度
        int class_id;    // EfficientNet分类结果 (0 or 1 for your case)
        // 根据需要，可以添加更多 EfficientNet 的特征向量等
        // float* features; // 如果需要返回特征向量
        // int feature_size;

        C_Point kps[17]; // YOLOv8 Pose通常有17个关键点
        int num_kps;     // 实际关键点数量
    } C_Extended_Person_Feats;

#ifdef __cplusplus
}
#endif


#endif //C_DSTRUCT_H
