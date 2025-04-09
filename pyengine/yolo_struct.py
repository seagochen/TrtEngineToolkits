import ctypes
from ctypes import c_int, c_float

# 定义结果结构体
# Define the structure for YoloStruct
class YoloStruct(ctypes.Structure):
    _fields_ = [
        ("lx", c_int),
        ("ly", c_int),
        ("rx", c_int),
        ("ry", c_int),
        ("conf", c_float),
        ("cls", c_int)
    ]


# Define the structure for YoloPointStruct
class YoloPointStruct(ctypes.Structure):
    _fields_ = [
        ("x", c_int),
        ("y", c_int),
        ("conf", c_float)
    ]