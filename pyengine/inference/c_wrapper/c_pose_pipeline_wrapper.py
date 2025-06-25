import numpy as np
import cv2
from c_pose_pipeline_ctypes import (
    lib,
    C_BatchedPoseResults,
    C_KeyPoint, C_YoloPose, C_ImagePoseResults
)

# 1) 注册模型（全局只需做一次）
lib.c_register_models()

# 2) 创建上下文
ctx = lib.c_create_pose_pipeline(
    b"/opt/models/yolov8n-pose.engine",
    b"/opt/models/efficientnet_b0_feat_logits.engine",
    8,    # yolo_max_batch
    32,   # efficient_max_batch
    0.4,  # cls threshold
    0.5   # iou threshold
)
if not ctx:
    raise RuntimeError("Failed to create YoloEfficientContext")

# 3) 准备一批 numpy 图像
paths = [f"/opt/images/supermarket/customer{i}.png" for i in range(1,9)]
imgs = [cv2.imread(p) for p in paths]
# HWC, uint8
batch = [cv2.resize(im, (640,640)) for im in imgs]

# 构造 C API 需要的原始指针数组
ptrs   = (POINTER(ctypes.c_ubyte) * len(batch))()
widths = (c_int * len(batch))()
heights= (c_int * len(batch))()
chans  = (c_int * len(batch))()
for i, im in enumerate(batch):
    if im is None:
        raise RuntimeError(f"failed to load {paths[i]}")
    arr = np.ascontiguousarray(im, dtype=np.uint8)
    ptrs[i]    = arr.ctypes.data_as(POINTER(ctypes.c_ubyte))
    heights[i] = arr.shape[0]
    widths[i]  = arr.shape[1]
    chans[i]   = arr.shape[2]

# 4) 调用推理
c_res: C_BatchedPoseResults = lib.c_process_batched_images(
    ctx,
    ptrs, widths, heights, chans,
    len(batch)
)

# 5) 把结果拷回 Python
py_results = []
for img_idx in range(c_res.num_images):
    imgres: C_ImagePoseResults = c_res.results[img_idx]
    dets = []
    for j in range(imgres.num_detections):
        yp: C_YoloPose = imgres.detections[j]
        # 把关键点拷进 Python list
        kpts = []
        for k in range(yp.num_pts):
            kp = yp.pts[k]
            kpts.append((kp.x, kp.y, kp.conf))
        dets.append({
            "bbox": (yp.lx, yp.ly, yp.rx, yp.ry),
            "score": yp.cls,
            "keypoints": kpts
        })
    py_results.append(dets)

# 6) 释放 C 端结果
lib.c_free_batched_pose_results(byref(c_res))

# 7) 最后清理上下文
lib.c_destroy_pose_pipeline(ctx)

# py_results 就是最终的 list[list[dict]] 结构
for i, dets in enumerate(py_results):
    print(f"Image {i} dete到 {len(dets)} 个检测框")
