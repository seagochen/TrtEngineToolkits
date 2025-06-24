import ctypes
from dataclasses import dataclass, field
from typing import List


class C_Point(ctypes.Structure):
    """
    C-style Point structure.
    typedef struct C_Point {
        float x;
        float y;
        float score; // 关键点置信度
    } C_Point;
    """
    _fields_ = [
        ('x', ctypes.c_float),
        ('y', ctypes.c_float),
        ('score', ctypes.c_float),
    ]

class C_Rect(ctypes.Structure):
    """
    C-style Bounding Box structure (x1, y1, x2, y2).
    typedef struct C_Rect {
        float x1;
        float y1;
        float x2;
        float y2;
    } C_Rect;
    """
    _fields_ = [
        ('x1', ctypes.c_float),
        ('y1', ctypes.c_float),
        ('x2', ctypes.c_float),
        ('y2', ctypes.c_float),
    ]

class C_Extended_Person_Feats(ctypes.Structure):
    """
    Corresponds to C++ YoloPose, but with added classification info.
    typedef struct C_Extended_Pose_Feats {
        // YOLO 检测结果
        C_Rect box;          // 边界框
        float confidence;    // 检测置信度
        float class_id;      // EfficientNet分类结果 (0 or 1 for your case)
        C_Point pts[17];     // YOLOv8 Pose通常有17个关键点

        // Additional human features
        float features[256]; // If feature vector needs to be returned
    } C_Extended_Person_Feats;
    """
    _fields_ = [
        ('box', C_Rect),
        ('confidence', ctypes.c_float),
        ('class_id', ctypes.c_float),
        ('pts', C_Point * 17),  # Array of 17 C_Point structures
        ('features', ctypes.c_float * 256), # Array of 256 floats
    ]

class C_Inference_Result(ctypes.Structure):
    """
    Final processing result for each image, encapsulates C_Extended_Pose_Feats.
    typedef struct C_Inference_Result {
        int num_detected;
        C_Extended_Person_Feats* detections;
    } C_Inference_Result;
    """
    _fields_ = [
        ('num_detected', ctypes.c_int),
        # Pointer to an array of C_Extended_Person_Feats
        ('detections', ctypes.POINTER(C_Extended_Person_Feats)),
    ]


@dataclass
class Point:
    """Pythonic representation of a 2D point with score."""
    x: float
    y: float
    score: float

@dataclass
class Rect:
    """Pythonic representation of a bounding box."""
    x1: float
    y1: float
    x2: float
    y2: float

@dataclass
class PoseDetection:
    """
    Pythonic representation of an individual pose detection,
    combining bounding box, confidence, class ID, keypoints, and features.
    """
    box: Rect
    confidence: float
    class_id: int # Often an integer for classification
    keypoints: List[Point] = field(default_factory=list) # List of Point objects
    features: List[float] = field(default_factory=list) # List of floats for the feature vector

@dataclass
class InferenceResult:
    """
    Pythonic representation of the full inference result for a single image,
    containing the number of detections and a list of detected objects.
    """
    num_detected: int
    detections: List[PoseDetection] = field(default_factory=list) # List of PoseDetection objects
