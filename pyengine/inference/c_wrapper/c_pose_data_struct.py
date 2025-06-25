import ctypes
from ctypes import c_int, c_float, c_char_p, c_void_p, POINTER, byref

# 对应 C_KeyPoint
class C_KeyPoint(ctypes.Structure):
    _fields_ = [
        ("x", c_float),
        ("y", c_float),
        ("conf", c_float),
    ]

# 对应 C_YoloPose
class C_YoloPose(ctypes.Structure):
    _fields_ = [
        ("lx",   c_int),
        ("ly",   c_int),
        ("rx",   c_int),
        ("ry",   c_int),
        ("cls",  c_float),
        ("num_pts", c_int),
        ("pts",  POINTER(C_KeyPoint)),
    ]

# 对应 C_ImagePoseResults
class C_ImagePoseResults(ctypes.Structure):
    _fields_ = [
        ("image_idx",    c_int),
        ("num_detections", c_int),
        ("detections",   POINTER(C_YoloPose)),
    ]

# 对应 C_BatchedPoseResults
class C_BatchedPoseResults(ctypes.Structure):
    _fields_ = [
        ("num_images", c_int),
        ("results",    POINTER(C_ImagePoseResults)),
    ]
