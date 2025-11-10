# TrtEngineToolkits

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20Jetson-green.svg)](https://developer.nvidia.com/embedded/jetson-developer-kits)
[![CUDA](https://img.shields.io/badge/CUDA-11.0%2B-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![TensorRT](https://img.shields.io/badge/TensorRT-8.0%2B-76B900.svg)](https://developer.nvidia.com/tensorrt)

## 概述

TrtEngineToolkits 是一个基于 NVIDIA TensorRT 的高性能推理工具包，提供易用的 C/C++ API 用于部署深度学习模型。项目支持多种视觉模型（YOLO、EfficientNet 等），并针对 x86 和 Jetson 平台进行了优化。

### 主要特性

- 🚀 **高性能推理**: 基于 TensorRT 优化的 GPU 加速
- 🎯 **多模型支持**: YOLO (检测、姿态)、EfficientNet (分类、特征提取)
- 🔧 **双版本架构**:
  - V1: 完整 C++ 实现，集成 OpenCV
  - V2: 纯 C API，无外部依赖，易于集成
- 🌐 **跨平台支持**: x86_64 和 ARM64 (Jetson)
- 📦 **易于集成**: 提供静态库和动态库
- 🐍 **Python 绑定**: 通过 pyengine 模块提供 Python 接口

## 目录结构

```
TrtEngineToolkits/
├── include/                    # 头文件
│   ├── trtengine/             # V1 版本 (C++)
│   └── trtengine_v2/          # V2 版本 (纯 C)
│       ├── core/              # TensorRT 引擎核心
│       ├── common/            # 通用数据结构和算法
│       ├── pipelines/         # 各种模型推理管线
│       │   ├── yolopose/     # YOLOv8-Pose 姿态检测
│       │   └── efficientnet/ # EfficientNet 分类/特征提取
│       ├── tensor/            # Tensor 操作
│       └── utils/             # 工具类
│
├── src/                       # 源代码
│   ├── trtengine_v2/         # V2 实现
│   ├── models/               # V1 模型实现
│   ├── c_apis/               # C API 封装
│   └── utils/                # 工具实现
│
├── examples/                  # 示例程序
│   ├── yolopose_pipeline_example.c
│   ├── efficientnet_pipeline_example.c
│   └── nms_example.c
│
├── scripts/                   # 构建和工具脚本
├── config/                    # 模型配置文件
├── pyengine/                  # Python 封装
└── CMakeLists.txt            # CMake 构建配置
```

## 快速开始

### 系统要求

#### 硬件要求
- NVIDIA GPU (计算能力 >= 6.0)
- 或 NVIDIA Jetson 开发板 (Nano, Xavier, Orin 等)

#### 软件依赖
- **必须**:
  - CMake >= 3.16
  - GCC >= 9.0 或 Clang >= 10.0
  - CUDA >= 11.0
  - TensorRT >= 8.0
  - SimpleCudaToolkits (需安装到 `/opt/SimpleCudaToolkits`)

- **可选** (仅 V1):
  - OpenCV >= 4.0
  - OpenMP

### 安装依赖

#### Ubuntu / Jetson

```bash
# 安装基础工具
sudo apt update
sudo apt install -y build-essential cmake git

# 安装 CUDA (如未安装)
# 参考: https://developer.nvidia.com/cuda-downloads

# 安装 TensorRT (如未安装)
# 参考: https://developer.nvidia.com/tensorrt

# 安装 OpenCV (可选，仅 V1 需要)
sudo apt install -y libopencv-dev

# 安装 SimpleCudaToolkits
# 从 https://github.com/your-repo/SimpleCudaToolkits 下载并安装
```

### 编译

#### V2 版本 (推荐 - 纯 C API)

```bash
# 克隆仓库
git clone https://github.com/your-org/TrtEngineToolkits.git
cd TrtEngineToolkits

# 配置并编译
cmake -B build -DBUILD_V2=ON
cmake --build build -j$(nproc)

# 编译结果
# build/lib/libjetson.so        - 动态库
# build/lib/libjetson.a         - 静态库
# build/examples/               - 示例程序
```

#### V1 版本 (传统 C++)

```bash
# 配置并编译
cmake -B build -DBUILD_V2=OFF
cmake --build build -j$(nproc)
```

### 运行示例

```bash
# 设置库路径
export LD_LIBRARY_PATH=$(pwd)/build/lib:$LD_LIBRARY_PATH

# 运行 YOLOv8-Pose 示例
./build/examples/yolopose_pipeline_example \
    /path/to/yolov8_pose.engine

# 运行 EfficientNet 示例
./build/examples/efficientnet_pipeline_example \
    /path/to/efficientnet.engine

# 运行 NMS 示例 (不需要模型)
./build/examples/nms_example
```

## 使用指南

### V2 API 使用 (推荐)

#### YOLOv8-Pose 姿态检测

```c
#include "trtengine_v2/pipelines/yolopose/c_yolopose_pipeline.h"

// 1. 创建配置
C_YoloPosePipelineConfig config = c_yolopose_pipeline_get_default_config();
config.engine_path = "/path/to/yolov8_pose.engine";
config.conf_threshold = 0.25f;
config.iou_threshold = 0.45f;

// 2. 创建 pipeline
C_YoloPosePipelineContext* pipeline = c_yolopose_pipeline_create(&config);

// 3. 准备输入图像 (RGB 格式)
C_ImageInput image = {
    .data = your_rgb_data,
    .width = 1920,
    .height = 1080,
    .channels = 3
};

// 4. 执行推理
C_YoloPoseImageResult result = {0};
c_yolopose_infer_single(pipeline, &image, &result);

// 5. 处理结果
printf("检测到 %zu 个人\n", result.num_poses);
for (size_t i = 0; i < result.num_poses; i++) {
    C_YoloPose* pose = &result.poses[i];
    printf("Person %zu: bbox=[%d,%d,%d,%d], conf=%.2f\n",
           i, pose->detection.lx, pose->detection.ly,
           pose->detection.rx, pose->detection.ry,
           pose->detection.conf);

    // 访问 17 个关键点
    for (int j = 0; j < 17; j++) {
        if (pose->pts[j].conf > 0.5f) {
            printf("  关键点 %d: (%.1f, %.1f)\n",
                   j, pose->pts[j].x, pose->pts[j].y);
        }
    }
}

// 6. 清理资源
c_yolopose_image_result_free(&result);
c_yolopose_pipeline_destroy(pipeline);
```

#### EfficientNet 分类和特征提取

```c
#include "trtengine_v2/pipelines/efficientnet/c_efficientnet_pipeline.h"

// 1. 创建配置
C_EfficientNetPipelineConfig config = c_efficientnet_pipeline_get_default_config();
config.engine_path = "/path/to/efficientnet.engine";

// 2. 创建 pipeline
C_EfficientNetPipelineContext* pipeline = c_efficientnet_pipeline_create(&config);

// 3. 准备输入
C_ImageInput image = {
    .data = your_rgb_data,
    .width = 640,
    .height = 480,
    .channels = 3
};

// 4. 执行推理
C_EfficientNetResult result = {0};
c_efficientnet_infer_single(pipeline, &image, &result);

// 5. 获取分类结果
printf("预测类别: %d\n", result.class_id);
printf("置信度: %.4f\n", result.confidence);

// 6. 获取特征向量 (256 维)
printf("特征向量:\n");
for (size_t i = 0; i < result.feature_size; i++) {
    printf("  [%zu]: %.4f\n", i, result.features[i]);
}

// 7. 清理
c_efficientnet_result_free(&result);
c_efficientnet_pipeline_destroy(pipeline);
```

### Python 使用 (通过 pyengine)

```python
from pyengine.inference import YoloPosePipeline

# 创建 pipeline
pipeline = YoloPosePipeline(
    engine_path="/path/to/yolov8_pose.engine",
    conf_threshold=0.25,
    iou_threshold=0.45
)

# 推理
import cv2
image = cv2.imread("image.jpg")
results = pipeline.infer(image)

# 处理结果
for i, pose in enumerate(results):
    print(f"Person {i}:")
    print(f"  BBox: {pose.bbox}")
    print(f"  Keypoints: {pose.keypoints}")
```

## 模型转换

### ONNX 转 TensorRT Engine

使用项目提供的脚本：

```bash
# YOLOv8-Pose
python scripts/build_engine.py \
    --onnx /path/to/yolov8n-pose.onnx \
    --output /path/to/yolov8n-pose.engine \
    --batch 1 \
    --workspace 4096

# EfficientNet
python scripts/build_engine.py \
    --onnx /path/to/efficientnet_b0.onnx \
    --output /path/to/efficientnet_b0.engine \
    --batch 8 \
    --workspace 2048
```

或使用配置文件：

```bash
# 使用 JSON 配置
python scripts/build_engine.py \
    --config config/efficientnet_feats.json
```

## 性能优化

### 批量推理

```c
// 批量处理可以显著提升吞吐量
config.max_batch_size = 8;

C_ImageBatch batch = {
    .count = 8,
    .images = images_array
};

C_YoloPoseBatchResult results = {0};
c_yolopose_infer_batch(pipeline, &batch, &results);
```

### 精度与速度权衡

```c
// 快速模式 (可能有误检)
config.conf_threshold = 0.15f;
config.input_width = 416;
config.input_height = 416;

// 精确模式 (可能漏检)
config.conf_threshold = 0.45f;
config.input_width = 1280;
config.input_height = 1280;

// 平衡模式 (推荐)
config.conf_threshold = 0.25f;
config.input_width = 640;
config.input_height = 640;
```

### 多线程推理

```c
// 每个线程创建独立的 pipeline 实例
void* inference_thread(void* arg) {
    ThreadData* data = (ThreadData*)arg;

    // 线程独立的 pipeline
    C_YoloPosePipelineContext* pipeline =
        c_yolopose_pipeline_create(&data->config);

    // 执行推理...
    c_yolopose_infer_single(pipeline, &data->image, &data->result);

    c_yolopose_pipeline_destroy(pipeline);
    return NULL;
}
```

## 应用场景

### 🏃 健身与体育
- 动作识别和计数 (深蹲、俯卧撑、引体向上)
- 姿势纠正和指导
- 运动轨迹分析
- 体能评估

### 🔒 安防监控
- 异常行为检测 (跌倒、打架、入侵)
- 人流统计和分析
- 危险姿势识别
- 区域入侵警报

### 🎮 人机交互
- 手势识别与控制
- 虚拟试衣与 AR
- 体感游戏
- 无接触控制

### 🏥 医疗健康
- 步态分析
- 康复训练监测
- 姿势评估
- 跌倒检测

### 🔍 图像检索
- 基于特征的相似图片搜索
- 人脸识别和验证
- 图像去重
- 内容推荐

## 性能基准

### NVIDIA Jetson Orin Nano

| 模型 | 输入尺寸 | Batch | FP16 | 延迟 (ms) | FPS |
|------|---------|-------|------|-----------|-----|
| YOLOv8n-Pose | 640x640 | 1 | ✓ | 15 | 66 |
| YOLOv8n-Pose | 640x640 | 4 | ✓ | 45 | 89 |
| EfficientNet-B0 | 224x224 | 1 | ✓ | 3 | 333 |
| EfficientNet-B0 | 224x224 | 8 | ✓ | 18 | 444 |

### NVIDIA RTX 3090

| 模型 | 输入尺寸 | Batch | FP16 | 延迟 (ms) | FPS |
|------|---------|-------|------|-----------|-----|
| YOLOv8n-Pose | 640x640 | 1 | ✓ | 2.5 | 400 |
| YOLOv8n-Pose | 640x640 | 16 | ✓ | 25 | 640 |
| EfficientNet-B0 | 224x224 | 1 | ✓ | 0.8 | 1250 |
| EfficientNet-B0 | 224x224 | 32 | ✓ | 15 | 2133 |

## 工具脚本

### Jetson 性能优化

```bash
# 设置最大性能模式
sudo ./scripts/jetson_power_clocks.sh --maxn
sudo ./scripts/jetson_power_clocks.sh --max-clocks

# 恢复正常模式
sudo ./scripts/jetson_power_clocks.sh --restore
```

### 监控工具安装

```bash
# Jetson: 安装 jtop
./scripts/install_pw_monitor.sh

# x86: 安装 nvitop
./scripts/install_pw_monitor.sh
```

## 常见问题

### Q: 编译时找不到 SimpleCudaToolkits？
A: 确保已安装到 `/opt/SimpleCudaToolkits`，或修改 CMakeLists.txt 中的路径。

### Q: 运行时提示找不到 libjetson.so？
A: 设置库路径：
```bash
export LD_LIBRARY_PATH=/path/to/build/lib:$LD_LIBRARY_PATH
```

### Q: Jetson 上性能不佳？
A: 确保使用了最大性能模式：
```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

### Q: 如何转换自己的模型？
A: 参考 `scripts/build_engine.py` 和 `config/` 目录下的示例配置。

### Q: 支持 INT8 量化吗？
A: 支持，在转换 engine 时指定 `--precision int8` 并提供校准数据。

### Q: 可以在 Windows 上使用吗？
A: 目前主要支持 Linux。Windows 支持需要修改部分路径和链接选项。

## 架构对比

### V1 vs V2

| 特性 | V1 | V2 |
|------|----|----|
| API 语言 | C++ | Pure C |
| OpenCV 依赖 | 需要 | 不需要 |
| 易于集成 | 中等 | 容易 |
| Python FFI | 复杂 | 简单 |
| 性能 | 高 | 高 |
| 维护成本 | 高 | 低 |
| 推荐用途 | 快速原型 | 生产部署 |

### 为什么选择 V2？

1. **无外部依赖**: 只需要 CUDA 和 TensorRT
2. **易于集成**: 纯 C API 可以从任何语言调用
3. **轻量级**: 更小的二进制大小
4. **稳定性**: 更少的依赖意味着更少的兼容性问题
5. **跨平台**: 更容易移植到其他平台

## 贡献指南

欢迎贡献！请遵循以下步骤：

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

### 代码规范

- C 代码遵循 Linux Kernel 风格
- C++ 代码遵循 Google C++ Style Guide
- 所有公开 API 必须有详细注释
- 添加新功能需要包含示例和测试

## 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

## 联系方式

- **作者**: TrtEngineToolkits Team
- **邮箱**: your-email@example.com
- **项目主页**: https://github.com/your-org/TrtEngineToolkits
- **问题反馈**: https://github.com/your-org/TrtEngineToolkits/issues

## 致谢

- NVIDIA TensorRT 团队
- Ultralytics (YOLOv8)
- SimpleCudaToolkits 项目

## 更新日志

### Version 2.0.0 (2025-11-10)
- ✨ 新增 V2 架构 (纯 C API)
- ✨ 新增 EfficientNet Pipeline
- ✨ 新增 YOLOv8-Pose Pipeline
- 🔧 重构代码结构 (common/pipelines)
- 📝 完善文档和示例
- 🚀 性能优化

### Version 1.0.0
- 🎉 初始版本发布
- ✅ 支持 YOLO 系列模型
- ✅ 支持 Jetson 和 x86 平台

---

⭐ 如果这个项目对你有帮助，请给个 Star！
