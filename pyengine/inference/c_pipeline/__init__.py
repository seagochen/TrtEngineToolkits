"""
C Pipeline Wrappers

This package provides Python wrappers for the TrtEngineToolkits C API.

V1 (Legacy):
- PosePipelineV2: Coupled YOLOv8-Pose + EfficientNet pipeline
- YoloPoseV2: YOLOv8-Pose wrapper (V1 API)

V2 (New Architecture):
- YoloPosePipelineV2: Standalone YOLOv8-Pose pipeline
- EfficientNetPipelineV2: Standalone EfficientNet pipeline
- c_structures_v2: Common C structure definitions
"""

# V1 Legacy pipelines
from .pipeline_v2 import PosePipelineV2
from .yolopose_v2 import YoloPoseV2

# V2 New architecture pipelines
from .yolopose_pipeline_v2 import YoloPosePipelineV2
from .efficientnet_pipeline_v2 import EfficientNetPipelineV2

# C structure definitions
from .c_structures_v2 import (
    C_KeyPoint,
    C_YoloDetect,
    C_ImageInput,
    C_YoloPose,
    C_EfficientNetResult,
    YOLO_POSE_NUM_KEYPOINTS,
    EFFICIENTNET_DEFAULT_FEAT_SIZE,
    EFFICIENTNET_DEFAULT_NUM_CLASSES,
    EFFICIENTNET_DEFAULT_IMAGE_SIZE
)

__all__ = [
    # V1 Legacy
    "PosePipelineV2",
    "YoloPoseV2",

    # V2 New architecture
    "YoloPosePipelineV2",
    "EfficientNetPipelineV2",

    # C structures
    "C_KeyPoint",
    "C_YoloDetect",
    "C_ImageInput",
    "C_YoloPose",
    "C_EfficientNetResult",
    "YOLO_POSE_NUM_KEYPOINTS",
    "EFFICIENTNET_DEFAULT_FEAT_SIZE",
    "EFFICIENTNET_DEFAULT_NUM_CLASSES",
    "EFFICIENTNET_DEFAULT_IMAGE_SIZE",
]
