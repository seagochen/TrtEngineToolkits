import math
from typing import List, Union

from pyengine.inference.yolo.data_struct import YoloPoseSorted, YoloPose


def angle_between(v1, v2):
    """计算两个二维向量 v1 和 v2 之间的夹角，返回角度（°）。"""
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    norm1 = math.hypot(v1[0], v1[1])
    norm2 = math.hypot(v2[0], v2[1])
    if norm1 == 0 or norm2 == 0:
        return 0.0
    # 防止浮点误差导致的超出[-1,1]范围
    cos_angle = max(min(dot / (norm1 * norm2), 1.0), -1.0)
    angle_rad = math.acos(cos_angle)
    return math.degrees(angle_rad)


def detect_pose_action(
        poses: List[Union['YoloPoseSorted', 'YoloPose']]
) -> List[int]:
    """
    根据传入的一组 YoloPoseSorted 或 YoloPose 对象，利用关键点信息
    判断人体姿态，返回每个对象对应的动作（“未知”、“下蹲”、“弯腰”、“站立”、“坐”）。
    0 - 未知
    1 - 弯腰
    2 - 坐
    3 - 下蹲
    4 - 站立
    """
    actions = []
    conf_threshold = 0.3  # 置信度阈值

    for pose in poses:
        pts = pose.pts
        if len(pts) != 17:
            actions.append(0) # 未知
            continue

        # 计算左右肩的中点
        shoulder_points = []
        if pts[5].conf >= conf_threshold:
            shoulder_points.append(pts[5])
        if pts[6].conf >= conf_threshold:
            shoulder_points.append(pts[6])
        if not shoulder_points:
            actions.append(0) # 未知
            continue
        mid_shoulder_x = sum(p.x for p in shoulder_points) / len(shoulder_points)
        mid_shoulder_y = sum(p.y for p in shoulder_points) / len(shoulder_points)

        # 计算左右臀（Hip）的中点（关键点索引11和12）
        hip_points = []
        if pts[11].conf >= conf_threshold:
            hip_points.append(pts[11])
        if pts[12].conf >= conf_threshold:
            hip_points.append(pts[12])
        if not hip_points:
            actions.append(0) # 未知
            continue
        mid_hip_x = sum(p.x for p in hip_points) / len(hip_points)
        mid_hip_y = sum(p.y for p in hip_points) / len(hip_points)

        # 计算躯干向量与垂直方向的夹角
        torso_vec = (mid_hip_x - mid_shoulder_x, mid_hip_y - mid_shoulder_y)
        vertical = (0, 1)  # 假定图像坐标系中 y 轴向下
        torso_angle = angle_between(torso_vec, vertical)

        # 计算左右侧膝关节的弯曲角度
        knee_angles = []
        # 右侧：使用关键点 11 (右Hip), 13 (右Knee), 15 (右Ankle)
        if pts[11].conf >= conf_threshold and pts[13].conf >= conf_threshold and pts[15].conf >= conf_threshold:
            right_hip = (pts[11].x, pts[11].y)
            right_knee = (pts[13].x, pts[13].y)
            right_ankle = (pts[15].x, pts[15].y)
            v1 = (right_knee[0] - right_hip[0], right_knee[1] - right_hip[1])
            v2 = (right_knee[0] - right_ankle[0], right_knee[1] - right_ankle[1])
            angle_right = angle_between(v1, v2)
            knee_angles.append(angle_right)
        # 左侧：使用关键点 12 (左Hip), 14 (左Knee), 16 (左Ankle)
        if pts[12].conf >= conf_threshold and pts[14].conf >= conf_threshold and pts[16].conf >= conf_threshold:
            left_hip = (pts[12].x, pts[12].y)
            left_knee = (pts[14].x, pts[14].y)
            left_ankle = (pts[16].x, pts[16].y)
            v1 = (left_knee[0] - left_hip[0], left_knee[1] - left_hip[1])
            v2 = (left_knee[0] - left_ankle[0], left_knee[1] - left_ankle[1])
            angle_left = angle_between(v1, v2)
            knee_angles.append(angle_left)

        if not knee_angles:
            actions.append(0)  # 未知
            continue
        avg_knee_angle = sum(knee_angles) / len(knee_angles)

        # 根据计算结果判断姿态：
        # 如果躯干明显前倾，则认为是“弯腰”
        if torso_angle > 30:
            actions.append(1)  # 弯腰
        else:
            # 根据膝关节角度判断：
            if avg_knee_angle < 110:
                actions.append(2) # 坐
            elif avg_knee_angle < 160:
                actions.append(3) # 下蹲
            else:
                actions.append(4) # 站立

    return actions