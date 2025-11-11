# Python V2 架构迁移指南

## 概述

本文档详细说明如何从 V1 架构迁移到 V2 架构。V2 架构的核心变化是**解耦**：YoloPose 和 EfficientNet 不再强制捆绑，可以独立使用。

## 主要变化总结

### 1. 架构变化

| 方面 | V1 (旧版) | V2 (新版) |
|------|-----------|-----------|
| 模型耦合 | YoloPose + EfficientNet 必须一起使用 | 完全解耦，可独立使用 |
| API 设计 | `PosePipelineV2` 单一类 | `YoloPosePipelineV2` + `EfficientNetPipelineV2` |
| features 位置 | 在 `Skeleton.features` | 在独立的 `ClassificationResult.features` |
| 后处理 | CPU | CUDA 加速 |
| C API | C++ 接口 | 纯 C 接口 |

### 2. 数据结构变化

#### ObjectDetection 和 Skeleton

**V1:**
```python
@dataclass
class ObjectDetection(InferenceResults):
    rect: Rect
    classification: int
    confidence: float
    track_id: int
    features: List[float]  # ❌ 包含在 ObjectDetection 中

@dataclass
class Skeleton(ObjectDetection):
    points: List[Point]
    # 继承了 features
```

**V2:**
```python
@dataclass
class ObjectDetection(InferenceResults):
    rect: Rect
    classification: int
    confidence: float
    track_id: int
    # ✅ 不再包含 features

@dataclass
class Skeleton(ObjectDetection):
    points: List[Point]
    # 不包含 features
```

#### 新增 ClassificationResult

**V2 新增:**
```python
@dataclass
class ClassificationResult(InferenceResults):
    """独立的分类结果类"""
    class_id: int
    confidence: float
    logits: List[float]
    features: List[float]  # ✅ features 现在在这里
```

### 3. API 变化

#### 导入变化

**V1:**
```python
from pyengine.inference.c_pipeline import PosePipelineV2, YoloPoseV2
from pyengine.inference.c_pipeline.converter import pipeline_v1_to_skeletons
```

**V2:**
```python
from pyengine.inference.c_pipeline import (
    YoloPosePipelineV2,
    EfficientNetPipelineV2,
    yolopose_to_skeletons,
    efficientnet_to_classifications,
    cascade_results_to_unified
)
from pyengine.inference.unified_structs import (
    Skeleton,
    ClassificationResult
)
```

## 迁移场景

### 场景 1: 只使用 YoloPose（不需要分类）

这是最简单的迁移场景。

**V1 (无法实现):**
```python
# V1 必须同时加载两个模型，即使不用 EfficientNet
pipeline = PosePipelineV2(
    library_path="...",
    yolo_engine_path="yolo.engine",
    efficient_engine_path="eff.engine",  # 必须提供
    ...
)
```

**V2 (推荐):**
```python
# 只加载 YoloPose
pipeline = YoloPosePipelineV2(
    library_path="libtrtengine_v2.so",
    engine_path="yolo.engine",
    conf_threshold=0.25,
    iou_threshold=0.45
)
pipeline.create()

# 推理
results = pipeline.infer([image_rgb])

# 转换为 Skeleton
from pyengine.inference.c_pipeline import yolopose_to_skeletons
skeletons_per_image = yolopose_to_skeletons(results)

# 使用
for skeletons in skeletons_per_image:
    for skeleton in skeletons:
        print(f"BBox: {skeleton.rect}")
        print(f"Keypoints: {len(skeleton.points)}")
        # skeleton 不再有 features 属性
```

### 场景 2: 只使用 EfficientNet（不需要姿态检测）

**V1 (无法实现):**
```python
# V1 无法单独使用 EfficientNet
```

**V2 (推荐):**
```python
# 只加载 EfficientNet
pipeline = EfficientNetPipelineV2(
    library_path="libtrtengine_v2.so",
    engine_path="eff.engine",
    num_classes=2,
    feature_size=512
)
pipeline.create()

# 推理
results = pipeline.infer([image_rgb])

# 转换为 ClassificationResult
from pyengine.inference.c_pipeline import efficientnet_to_classifications
classifications = efficientnet_to_classifications(results)

# 使用
for cls_result in classifications:
    print(f"Class: {cls_result.class_id}")
    print(f"Confidence: {cls_result.confidence}")
    print(f"Features: {cls_result.features[:10]}")  # ✅ features 在这里
```

### 场景 3: 级联使用（先检测，再分类）

这是完整的工作流程，等价于 V1 的功能。

**V1:**
```python
# 创建耦合的 pipeline
pipeline = PosePipelineV2(
    library_path="...",
    yolo_engine_path="yolo.engine",
    efficient_engine_path="eff.engine",
    ...
)
pipeline.register()
pipeline.create()

# 推理（自动级联）
results = pipeline.infer([image])

# 转换
from pyengine.inference.c_pipeline.converter import pipeline_v1_to_skeletons
skeletons_per_image = pipeline_v1_to_skeletons(results)

# skeletons 包含 features
for skeletons in skeletons_per_image:
    for skeleton in skeletons:
        print(f"Features: {skeleton.features}")  # ✅ V1 有 features
```

**V2:**
```python
# 创建两个独立的 pipeline
yolo_pipeline = YoloPosePipelineV2(
    library_path="libtrtengine_v2.so",
    engine_path="yolo.engine"
)
eff_pipeline = EfficientNetPipelineV2(
    library_path="libtrtengine_v2.so",
    engine_path="eff.engine"
)

yolo_pipeline.create()
eff_pipeline.create()

# 1. YoloPose 检测
yolo_results = yolo_pipeline.infer([image])
skeletons_per_image = yolopose_to_skeletons(yolo_results)

# 2. 对每个检测区域进行分类
classifications_per_detection = {}

for img_idx, skeletons in enumerate(skeletons_per_image):
    for det_idx, skeleton in enumerate(skeletons):
        # 裁剪区域
        bbox = skeleton.rect
        crop = image[int(bbox.y1):int(bbox.y2), int(bbox.x1):int(bbox.x2)]

        # EfficientNet 推理
        eff_results = eff_pipeline.infer([crop])
        classifications = efficientnet_to_classifications(eff_results)

        # 存储结果
        classifications_per_detection[(img_idx, det_idx)] = classifications[0]

# 3. 合并结果
for img_idx, skeletons in enumerate(skeletons_per_image):
    for det_idx, skeleton in enumerate(skeletons):
        key = (img_idx, det_idx)
        if key in classifications_per_detection:
            cls_result = classifications_per_detection[key]
            print(f"Person {det_idx}:")
            print(f"  BBox: {skeleton.rect}")
            print(f"  Class: {cls_result.class_id}")
            print(f"  Features: {cls_result.features[:10]}")  # ✅ features 单独存储
```

**V2 (使用辅助函数):**
```python
# 使用 cascade 辅助函数简化
from pyengine.inference.c_pipeline import cascade_results_to_unified

unified_results = cascade_results_to_unified(
    yolopose_results=yolo_results,
    efficientnet_results_per_detection=classifications_per_detection
)

# unified_results 格式:
# [
#     [  # 图像 0
#         {
#             'skeleton': Skeleton(...),
#             'classification': ClassificationResult(...)
#         },
#         ...
#     ],
#     ...
# ]
```

### 场景 4: 使用 Tracker（需要 features）

Tracker 已更新以支持 V2 架构。

**V1:**
```python
from pyengine.algorithms.tracker.tracker import UnifiedTrack

# detection 包含 features
track = UnifiedTrack(detection, use_reid=True)
track.update(new_detection)  # features 自动提取
```

**V2:**
```python
from pyengine.algorithms.tracker.tracker import UnifiedTrack

# Skeleton 不包含 features，需要单独提供
skeleton = ...  # Skeleton 对象
cls_result = ...  # ClassificationResult 对象

# 创建 track（传入 skeleton，features 可选）
track = UnifiedTrack(skeleton, use_reid=True)

# 更新 track（需要单独传入 features）
track.update(new_skeleton, features=cls_result.features if cls_result else None)
```

**兼容模式（如果不用 Re-ID）:**
```python
# 不使用 Re-ID features
track = UnifiedTrack(skeleton, use_reid=False)
track.update(new_skeleton)  # 不需要 features
```

## 可视化模块

可视化模块 (`pyengine.visualization`) **无需修改**，因为：
- `InferenceDrawer` 使用的是 `Skeleton` 和 `ExpandedSkeleton`
- 这些类的可视化相关属性（`rect`, `points`）没有变化
- `features` 属性只是被移除，不影响绘制

**使用示例:**
```python
from pyengine.visualization import InferenceDrawer
from pyengine.visualization.scheme_loader import SchemeLoader

drawer = InferenceDrawer(
    scheme_loader=SchemeLoader("config/scheme.json")
)

# V1 和 V2 用法相同
frame_with_results = drawer.draw_inference(
    frame,
    skeletons,
    draw_bbox=True,
    draw_keypoints=True,
    draw_skeleton_links=True
)
```

## 算法模块

### Filters (无需修改)

滤波器模块 (`pyengine.algorithms.filters`) 工作在数值层面，不依赖数据结构：

```python
from pyengine.algorithms.filters.seq_filters import apply_savgol_filter_1d

# V1 和 V2 用法相同
smoothed = apply_savgol_filter_1d(trajectory, window=5, polyorder=2)
```

### Estimation (无需修改)

姿态估计模块 (`pyengine.algorithms.estimation`) 使用 `Skeleton` 和 `ExpandedSkeleton`，接口未变：

```python
from pyengine.algorithms.estimation.posture_simple_calculation import (
    calculate_direction_and_posture
)

# V1 和 V2 用法相同
expanded_skeleton = calculate_direction_and_posture(skeleton)
print(f"Posture: {expanded_skeleton.posture_type}")
print(f"Direction: {expanded_skeleton.direction_type}")
```

## 常见问题 (FAQ)

### Q1: 我的代码访问了 `skeleton.features`，怎么办？

**A:** V2 中 `Skeleton` 不再包含 `features`。你有两个选择：

1. **如果不需要 features**: 删除相关代码
2. **如果需要 features**: 使用级联模式，单独获取 `ClassificationResult`

```python
# 旧代码
features = skeleton.features  # ❌ V2 中不存在

# 新代码 (选项 1: 不使用 features)
# 删除这行代码

# 新代码 (选项 2: 使用级联模式)
cls_result = classifications_per_detection[(img_idx, det_idx)]
features = cls_result.features  # ✅ 从 ClassificationResult 获取
```

### Q2: Tracker 怎么使用 features？

**A:** 更新 `track.update()` 调用，单独传入 features：

```python
# V1
track.update(detection)  # features 自动提取

# V2
track.update(skeleton, features=cls_result.features if cls_result else None)
```

### Q3: 我能同时用 V1 和 V2 吗？

**A:** 不建议。V1 封装文件已删除：
- ~~`pipeline_v2.py`~~
- ~~`yolopose_v2.py`~~
- ~~`detect_base.py`~~
- ~~`converter.py`~~

如果需要渐进迁移，建议：
1. 先迁移不依赖 features 的模块
2. 然后迁移级联推理部分
3. 最后迁移 tracking 部分

### Q4: 性能会受影响吗？

**A:** V2 性能更好：
- YoloPose 后处理使用 CUDA (5-10x 加速)
- 按需加载模型，降低内存占用
- 解耦设计允许并行推理

### Q5: 向后兼容性如何？

**A:** 提供了部分向后兼容：
- `pipeline_v2_to_skeletons()` 是 `yolopose_to_skeletons()` 的别名
- `Skeleton` 接口基本不变（除了 features）
- 可视化和算法模块无需修改

## 迁移检查清单

- [ ] 更新导入语句
- [ ] 替换 Pipeline 类
  - [ ] `PosePipelineV2` → `YoloPosePipelineV2` + `EfficientNetPipelineV2`
  - [ ] `YoloPoseV2` → `YoloPosePipelineV2`
- [ ] 移除对 `skeleton.features` 的访问
- [ ] 更新 Tracker 调用（如果使用 Re-ID）
- [ ] 测试可视化功能
- [ ] 测试算法模块
- [ ] 性能测试

## 参考示例

完整的示例代码位于：
- `examples_v2_python/yolopose_standalone_example.py`
- `examples_v2_python/efficientnet_standalone_example.py`
- `examples_v2_python/cascade_example.py`

## 技术支持

如果遇到问题：
1. 查看 `PYTHON_V2_WRAPPERS.md` 了解 API 详情
2. 查看示例代码 `examples_v2_python/`
3. 查看 C API 头文件 `include/trtengine_v2/pipelines/`

## 总结

V2 架构提供了更灵活、更高效的接口。虽然需要一些代码修改，但带来的好处包括：
- ✅ 解耦设计，按需使用
- ✅ CUDA 加速，性能提升
- ✅ 纯 C API，跨语言兼容
- ✅ 更清晰的数据结构
- ✅ 更低的内存占用

建议所有新项目直接使用 V2 架构！
