//
// Created by user on 6/16/25.
//

#ifndef SCT_YOLO_POST_PROCESS_H
#define SCT_YOLO_POST_PROCESS_H

#include <vector>

/**
 *　@brief YOLO系列模型的后处理函数，其过程基本上是通用的。
 * 最大的区别仅仅在于姿态估计时, 不执行分类处理。
 *
 * @param ptr_device 指向CUDA输出结果的指针
 * @param output 存储处理后的结果
 * @param features 特征数量
 * @param samples 样本数量
 * @param cls 分类阈值
 * @param use_pose 是否使用姿态估计
 * @return 返回处理后的结果数量
 */
int sct_yolo_post_proc(const float* ptr_device, std::vector<float>& output,
                                 int features, int samples, float cls, bool use_pose = false);

#endif //SCT_YOLO_POST_PROCESS_H
