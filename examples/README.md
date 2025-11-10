# TrtEngine V2 Examples

## 概述

这个目录包含了 TrtEngine V2 的各种示例程序，展示如何使用不同的 pipeline。

## 编译

当使用 `BUILD_V2=ON` 选项编译项目时，所有 examples 目录下的 `.c` 和 `.cpp` 文件会自动编译成独立的可执行文件。

### 编译步骤

```bash
# 清理旧的构建
rm -rf build

# 使用 V2 模式编译
cmake -B build -DBUILD_V2=ON

# 构建项目（包括所有 examples）
cmake --build build

# 或者使用多核并行编译
cmake --build build -j$(nproc)
```

### 输出位置

所有编译好的示例程序会输出到 `build/examples/` 目录：

```
build/
└── examples/
    ├── efficientnet_pipeline_example
    ├── yolopose_pipeline_example
    └── nms_example
```

## 可用示例

### 1. YOLOv8-Pose Pipeline Example

**文件**: `yolopose_pipeline_example.c`

**功能**: 演示如何使用 YOLOv8-Pose 进行人体姿态检测

**运行**:
```bash
./build/examples/yolopose_pipeline_example /path/to/yolov8_pose.engine
```

**示例内容**:
- 单张图片推理
- 批量图片推理
- 关键点可视化
- NMS 后处理

### 2. EfficientNet Pipeline Example

**文件**: `efficientnet_pipeline_example.c`

**功能**: 演示如何使用 EfficientNet 进行图像分类和特征提取

**运行**:
```bash
./build/examples/efficientnet_pipeline_example /path/to/efficientnet.engine
```

**示例内容**:
- 单张图片分类
- 批量图片推理
- 特征向量提取
- 特征相似度计算

### 3. NMS Example

**文件**: `nms_example.c`

**功能**: 演示如何使用通用的 NMS (非极大值抑制) 算法

**运行**:
```bash
./build/examples/nms_example
```

**示例内容**:
- 基础 NMS 操作
- IoU 计算
- 检测框过滤

## 添加新示例

要添加新的示例程序：

1. 在 `examples/` 目录下创建新的 `.c` 或 `.cpp` 文件
2. 包含必要的头文件
3. 实现 `main()` 函数
4. 重新运行 CMake 配置

**示例模板**:

```c
// my_example.c
#include "trtengine_v2/pipelines/your_pipeline/your_pipeline.h"
#include <stdio.h>

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <engine_path>\n", argv[0]);
        return 1;
    }

    // 你的代码...

    return 0;
}
```

重新配置：
```bash
cmake -B build -DBUILD_V2=ON
cmake --build build
```

新的示例会自动出现在 `build/examples/my_example`

## 依赖关系

所有示例程序会自动链接：
- `jetson_shared`: 主库 (包含所有 V2 pipelines)
- CUDA runtime
- TensorRT
- 数学库 (libm)

## 调试

如果需要调试示例程序：

```bash
# 编译时启用调试符号
cmake -B build -DBUILD_V2=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build

# 使用 gdb 调试
gdb ./build/examples/yolopose_pipeline_example
```

## 性能测试

要测试推理性能：

```bash
# 使用 time 命令
time ./build/examples/efficientnet_pipeline_example /path/to/model.engine

# 或使用 nvidia-smi 监控 GPU 使用
watch -n 0.1 nvidia-smi
```

## 常见问题

### Q: 为什么编译时找不到示例？
A: 确保使用了 `BUILD_V2=ON` 选项：
```bash
cmake -B build -DBUILD_V2=ON
```

### Q: 运行时提示找不到库？
A: 设置 LD_LIBRARY_PATH：
```bash
export LD_LIBRARY_PATH=/path/to/build/lib:$LD_LIBRARY_PATH
./build/examples/your_example
```

### Q: 如何只编译特定示例？
A: 使用 CMake target：
```bash
cmake --build build --target yolopose_pipeline_example
```

### Q: 示例程序占用内存过大？
A: 检查：
1. batch_size 设置是否过大
2. 模型输入分辨率是否过高
3. 是否正确释放了内存资源

## 贡献

欢迎贡献新的示例程序！请确保：
1. 代码清晰易懂
2. 包含适当的注释
3. 提供使用说明
4. 正确处理错误和资源释放

## 许可证

与主项目保持一致

## 作者
TrtEngineToolkits

## 更新日期
2025-11-10
