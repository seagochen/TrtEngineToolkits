# V2 级联推理测试程序

## 概述

`functional_test_v2_cascade` 是一个完整的 V2 级联推理测试程序，演示了如何将 YOLOv8-Pose 和 EfficientNet 两个 pipeline 串联使用，实现从人物检测到特征提取的完整流程。

## 功能流程

```
输入图片
    ↓
[1] YOLOv8-Pose 检测
    ├─ 检测人物边界框
    └─ 检测 17 个关键点
    ↓
[2] 可视化
    ├─ 绘制边界框
    ├─ 绘制关键点 (彩色圆点)
    └─ 绘制骨架连线
    ↓
[3] 保存可视化结果
    └─ output_pose.jpg
    ↓
[4] 裁剪人物区域
    ├─ 根据边界框裁剪
    └─ 保存裁剪图像
    ↓
[5] EfficientNet 分类
    ├─ 输入: 裁剪的人物图像
    ├─ 输出: 分类结果 (class_id, confidence)
    └─ 输出: 特征向量 (256 维)
    ↓
[6] 输出结果
```

## 编译

```bash
# 1. 配置 V2 版本
cmake -B build -DBUILD_V2=ON

# 2. 编译项目
cmake --build build -j$(nproc)

# 编译结果:
# build/functional_test_v2_cascade
```

## 使用方法

### 基本用法

```bash
./build/functional_test_v2_cascade \
    <yolopose_engine> \
    <efficientnet_engine> \
    <input_image>
```

### 参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| `yolopose_engine` | YOLOv8-Pose TensorRT 引擎路径 | `yolov8n-pose.engine` |
| `efficientnet_engine` | EfficientNet TensorRT 引擎路径 | `efficientnet_b0.engine` |
| `input_image` | 输入图片路径 (支持 jpg, png 等) | `person.jpg` |

### 完整示例

```bash
# 设置库路径
export LD_LIBRARY_PATH=$(pwd)/build/lib:$LD_LIBRARY_PATH

# 运行测试
./build/functional_test_v2_cascade \
    /opt/models/yolov8n-pose.engine \
    /opt/models/efficientnet_b0_feat_logits.engine \
    test_images/people.jpg
```

## 输出说明

### 1. 终端输出

程序会在终端输出详细的处理信息：

```
========================================
V2 级联推理测试程序
========================================
YOLOv8-Pose Engine: yolov8n-pose.engine
EfficientNet Engine: efficientnet_b0.engine
Input Image: people.jpg
========================================

[1/6] 加载输入图像...
  图像尺寸: 1920x1080, 通道数: 3

[2/6] 创建 YOLOv8-Pose Pipeline...
  Pipeline 创建成功

[3/6] 运行姿态检测...
  检测到 3 个人
  Person 0: bbox=[245,120,580,890], conf=0.92
  Person 1: bbox=[720,150,1045,920], conf=0.88
  Person 2: bbox=[1250,180,1580,850], conf=0.85
  已保存姿态图像: output_pose.jpg

[4/6] 创建 EfficientNet Pipeline...
  Pipeline 创建成功

[5/6] 裁剪人物区域并进行分类...

--- Person 0 ---
  裁剪区域: 345x780
  已保存裁剪图像: output_crop_0.jpg
  分类结果:
    预测类别: 1
    置信度: 2.3456
    Logits: -1.2345 2.3456
  特征向量信息:
    维度: 256
    L2 范数: 12.3456
    前 10 个特征值: 0.1234 -0.5678 0.9012 ...

--- Person 1 ---
  裁剪区域: 335x780
  已保存裁剪图像: output_crop_1.jpg
  ...

[6/6] 清理资源...
  完成

========================================
处理完成！
========================================
输出文件:
  - output_pose.jpg       : 带关键点和骨架的图像
  - output_crop_*.jpg     : 裁剪的人物区域
========================================
```

### 2. 输出文件

| 文件名 | 说明 | 内容 |
|--------|------|------|
| `output_pose.jpg` | 可视化结果 | 原图 + 边界框 + 关键点 + 骨架连线 |
| `output_crop_0.jpg` | 第 1 个人的裁剪区域 | 根据边界框裁剪的图像 |
| `output_crop_1.jpg` | 第 2 个人的裁剪区域 | 同上 |
| `output_crop_N.jpg` | 第 N+1 个人的裁剪区域 | 同上 |

## 可视化说明

### 关键点颜色编码

程序使用不同颜色标识不同类型的关键点：

| 颜色 | 关键点 | 索引 |
|------|--------|------|
| 🔴 红色 | 鼻子 | 0 |
| 🔵 蓝色 | 眼睛和耳朵 | 1-4 |
| 🟢 绿色 | 肩膀 | 5-6 |
| 🔷 青色 | 手臂 (肘、腕) | 7-10 |
| 🟣 品红 | 腿部 (髋、膝、踝) | 11-16 |

### 骨架连线

程序会绘制以下连线（黄色）：
- 鼻子 ↔ 眼睛
- 眼睛 ↔ 耳朵
- 鼻子 ↔ 肩膀
- 肩膀 ↔ 肘 ↔ 腕 (左右手臂)
- 肩膀 ↔ 髋 (躯干)
- 髋 ↔ 膝 ↔ 踝 (左右腿)

### 边界框

- 绿色矩形框标识检测到的人物区域

## 技术细节

### 图像处理

程序使用 OpenCV 进行图像处理：

1. **图像加载**: 使用 `cv::imread()` 加载各种格式图像
2. **图像保存**: 使用 `cv::imwrite()` 保存图像
3. **绘制圆形**: 使用 `cv::circle()` 绘制关键点
4. **绘制直线**: 使用 `cv::line()` 绘制骨架连线
5. **绘制矩形**: 使用 `cv::rectangle()` 绘制边界框
6. **图像裁剪**: 使用 `cv::Mat` 的 ROI 提取人物区域

### COCO 关键点定义

17 个关键点按照 COCO 格式定义：

```
0:  鼻子 (Nose)
1:  左眼 (Left Eye)
2:  右眼 (Right Eye)
3:  左耳 (Left Ear)
4:  右耳 (Right Ear)
5:  左肩 (Left Shoulder)
6:  右肩 (Right Shoulder)
7:  左肘 (Left Elbow)
8:  右肘 (Right Elbow)
9:  左腕 (Left Wrist)
10: 右腕 (Right Wrist)
11: 左髋 (Left Hip)
12: 右髋 (Right Hip)
13: 左膝 (Left Knee)
14: 右膝 (Right Knee)
15: 左踝 (Left Ankle)
16: 右踝 (Right Ankle)
```

### 骨架连接

19 条骨架连线：

```c
{0,1}, {0,2},     // 鼻子到眼睛
{1,3}, {2,4},     // 眼睛到耳朵
{0,5}, {0,6},     // 鼻子到肩膀
{5,7}, {7,9},     // 左臂
{6,8}, {8,10},    // 右臂
{5,6},            // 肩膀连接
{5,11}, {6,12},   // 肩膀到髋部
{11,12},          // 髋部连接
{11,13}, {13,15}, // 左腿
{12,14}, {14,16}  // 右腿
```

## 应用场景

### 1. 安防监控
- 检测并识别进入特定区域的人员
- 提取人员特征用于身份验证
- 异常姿态检测

### 2. 零售分析
- 顾客姿态分析
- 提取顾客特征用于分析
- 人流统计

### 3. 健身应用
- 动作识别
- 用户身份识别
- 个性化训练

### 4. 图像检索
- 基于姿态和特征的人物搜索
- 相似人物查找
- 人物聚类

## 性能优化

### 提高速度

```bash
# 1. 降低输入分辨率
# 编辑代码，修改:
# pose_config.input_width = 416;
# pose_config.input_height = 416;

# 2. 降低置信度阈值 (可能增加误检)
# pose_config.conf_threshold = 0.15f;

# 3. 使用 FP16 或 INT8 精度的 engine
```

### 提高精度

```bash
# 1. 使用更高分辨率
# pose_config.input_width = 1280;
# pose_config.input_height = 1280;

# 2. 提高置信度阈值
# pose_config.conf_threshold = 0.45f;

# 3. 使用更大的模型 (yolov8m-pose, yolov8l-pose)
```

## 故障排查

### Q: 编译时找不到 OpenCV？
A: 确保系统已安装 OpenCV 库，并且 CMake 可以找到它。

### Q: 运行时提示找不到 libjetson.so？
A: 设置库路径：
```bash
export LD_LIBRARY_PATH=/path/to/build/lib:$LD_LIBRARY_PATH
```

### Q: 检测不到人物？
A: 检查：
1. 图像中是否有足够清晰的人物
2. YOLOv8-Pose 引擎是否正确
3. 降低 conf_threshold

### Q: 分类结果不准确？
A: 检查：
1. EfficientNet 模型是否适合你的任务
2. 裁剪的图像质量
3. 模型是否需要重新训练

## 扩展建议

### 1. 添加多线程处理

```c
// 使用 OpenMP 并行处理多个人物
#pragma omp parallel for
for (size_t i = 0; i < pose_result.num_poses; i++) {
    // 处理每个人...
}
```

### 2. 添加视频支持

```cpp
// 使用 OpenCV 读取视频帧
cv::VideoCapture cap("video.mp4");
cv::Mat frame;
while (cap.read(frame)) {
    // 处理每一帧...
}
```

### 3. 添加特征匹配

```c
// 计算两个特征向量的余弦相似度
float cosine_similarity(float* feat1, float* feat2, size_t size) {
    float dot = 0.0f, norm1 = 0.0f, norm2 = 0.0f;
    for (size_t i = 0; i < size; i++) {
        dot += feat1[i] * feat2[i];
        norm1 += feat1[i] * feat1[i];
        norm2 += feat2[i] * feat2[i];
    }
    return dot / (sqrtf(norm1) * sqrtf(norm2));
}
```

### 4. 添加特征数据库

```c
// 保存特征到数据库
save_feature_to_db(person_id, feature_vector, feature_size);

// 搜索相似特征
person_id = search_similar_feature(query_feature, threshold);
```

## 参考资源

- [YOLOv8-Pose 文档](src/trtengine_v2/pipelines/yolopose/README.md)
- [EfficientNet 文档](src/trtengine_v2/pipelines/efficientnet/README.md)
- [COCO 关键点格式](https://cocodataset.org/#keypoints-2020)
- [OpenCV 文档](https://docs.opencv.org/)

## 作者
TrtEngineToolkits

## 更新日期
2025-11-10
