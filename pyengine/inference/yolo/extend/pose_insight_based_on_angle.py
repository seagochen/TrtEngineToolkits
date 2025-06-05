import math
from dataclasses import dataclass
from typing import List, Union, Tuple

import numpy as np
from pyengine.inference.yolo.d_struct.data_struct import YoloPose, YoloPoseSorted
from pyengine.inference.yolo.extend.basic import Posture, FacialDirection


# -------------- 核心分析类 --------------
class PoseInsight:
    """
    PoseInsight 类整合了人体姿态与面部朝向的分析。
    调用 analyze_poses() 方法，传入包含 YoloPose 或 YoloPoseSorted 对象的列表，
    返回每个 pose 的 Posture 和 FacialDirection 信息。
    """
    # 定义面部方向映射
    ORIENTATION_TEXTS = {
        0: "Front",   # 正面
        1: "Left",    # 左侧
        2: "Right",   # 右侧
        3: "Back",    # 背面
        -1: "Unknown"
    }

    @staticmethod
    def _angle_between(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
        """计算二维向量 v1 和 v2 之间的夹角（单位：度）"""
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        norm1 = math.hypot(v1[0], v1[1])
        norm2 = math.hypot(v2[0], v2[1])
        if norm1 == 0 or norm2 == 0:
            return 0.0
        # 防止浮点数误差超出 [-1, 1]
        cos_angle = max(min(dot / (norm1 * norm2), 1.0), -1.0)
        angle_rad = math.acos(cos_angle)
        return math.degrees(angle_rad)

    @staticmethod
    def _analyze_body_pose(pose: Union[YoloPose, YoloPoseSorted]) -> Posture:
        """
        根据 pose 中的关键点信息判断人体姿态，返回 Posture 对象，编码规则如下：
          0 - 未知
          1 - 弯腰
          2 - 坐
          3 - 下蹲
          4 - 站立
        """
        conf_threshold = 0.3
        pts = pose.pts
        if len(pts) != 17:
            return Posture(action=0)  # 未知

        # 计算左右肩中点（索引 5 和 6）
        shoulder_points = []
        if pts[5].conf >= conf_threshold:
            shoulder_points.append(pts[5])
        if pts[6].conf >= conf_threshold:
            shoulder_points.append(pts[6])
        if not shoulder_points:
            return Posture(action=0)
        mid_shoulder_x = sum(p.x for p in shoulder_points) / len(shoulder_points)
        mid_shoulder_y = sum(p.y for p in shoulder_points) / len(shoulder_points)

        # 计算左右臀中点（索引 11 和 12）
        hip_points = []
        if pts[11].conf >= conf_threshold:
            hip_points.append(pts[11])
        if pts[12].conf >= conf_threshold:
            hip_points.append(pts[12])
        if not hip_points:
            return Posture(action=0)
        mid_hip_x = sum(p.x for p in hip_points) / len(hip_points)
        mid_hip_y = sum(p.y for p in hip_points) / len(hip_points)

        # 躯干向量与垂直方向的夹角（图像坐标系中 y 轴向下）
        torso_vec = (mid_hip_x - mid_shoulder_x, mid_hip_y - mid_shoulder_y)
        vertical = (0, 1)
        torso_angle = PoseInsight._angle_between(torso_vec, vertical)

        # 计算左右膝关节弯曲角度
        knee_angles = []
        # 右侧：索引 11 (Hip), 13 (Knee), 15 (Ankle)
        if pts[11].conf >= conf_threshold and pts[13].conf >= conf_threshold and pts[15].conf >= conf_threshold:
            right_hip = (pts[11].x, pts[11].y)
            right_knee = (pts[13].x, pts[13].y)
            right_ankle = (pts[15].x, pts[15].y)
            v1 = (right_knee[0] - right_hip[0], right_knee[1] - right_hip[1])
            v2 = (right_knee[0] - right_ankle[0], right_knee[1] - right_ankle[1])
            knee_angles.append(PoseInsight._angle_between(v1, v2))
        # 左侧：索引 12 (Hip), 14 (Knee), 16 (Ankle)
        if pts[12].conf >= conf_threshold and pts[14].conf >= conf_threshold and pts[16].conf >= conf_threshold:
            left_hip = (pts[12].x, pts[12].y)
            left_knee = (pts[14].x, pts[14].y)
            left_ankle = (pts[16].x, pts[16].y)
            v1 = (left_knee[0] - left_hip[0], left_knee[1] - left_hip[1])
            v2 = (left_knee[0] - left_ankle[0], left_knee[1] - left_ankle[1])
            knee_angles.append(PoseInsight._angle_between(v1, v2))

        if not knee_angles:
            return Posture(action=0)
        avg_knee_angle = sum(knee_angles) / len(knee_angles)

        # 根据躯干前倾和膝关节角度进行判断
        if torso_angle > 30:
            return Posture(action=1)  # 弯腰
        else:
            if avg_knee_angle < 110:
                return Posture(action=2)  # 坐
            elif avg_knee_angle < 160:
                return Posture(action=3)  # 下蹲
            else:
                return Posture(action=4)  # 站立

    @staticmethod
    def _is_valid_point(pt) -> bool:
        """判断关键点是否有效：置信度 > 0.2 且坐标均大于 0"""
        return pt.conf > 0.2 and pt.x > 0 and pt.y > 0

    @staticmethod
    def _compute_modulus(pose: Union[YoloPose, YoloPoseSorted]) -> int:
        """
        根据检测框宽度计算模长，若检测框有效则返回宽度的 1/3，
        否则返回默认值 10。
        """
        if pose.lx > 0 and pose.ly > 0:
            return int(abs(pose.lx - pose.rx) / 3.0)
        return 10

    @staticmethod
    def _analyze_facial_direction(pose: Union[YoloPose, YoloPoseSorted]) -> FacialDirection:
        """
        根据 pose 中的面部关键点信息分析面部朝向，返回 FacialDirection 对象。
        逻辑：
          1. 当鼻子 + 双眼有效时：
             - 计算眼中点到鼻子的向量，得出角度
             - 如果左右耳均有效，则使用 (鼻子 - 耳中点) 归一化得到最终方向
          2. 当鼻子与仅一只眼有效时，直接判断左右侧
          3. 当脸部关键点不可用，但双耳有效时，判断为背面
          4. 否则判定为未知，原点取检测框左上角，方向向量为 (0, 0)
        """
        modulus = PoseInsight._compute_modulus(pose)
        pts = pose.pts

        # 按 YOLOPOSE 顺序：鼻子, 右眼, 左眼, 右耳, 左耳
        nose = pts[0]
        right_eye = pts[1]
        left_eye = pts[2]
        right_ear = pts[3]
        left_ear = pts[4]

        valid_nose = PoseInsight._is_valid_point(nose)
        valid_right_eye = PoseInsight._is_valid_point(right_eye)
        valid_left_eye = PoseInsight._is_valid_point(left_eye)
        valid_right_ear = PoseInsight._is_valid_point(right_ear)
        valid_left_ear = PoseInsight._is_valid_point(left_ear)

        # 默认初始值
        orientation = -1
        vec_x, vec_y = 0.0, 0.0
        origin_x, origin_y = int(pose.lx), int(pose.ly)

        # 情况1：鼻子和双眼均有效
        if valid_nose and valid_left_eye and valid_right_eye:
            mid_x = (left_eye.x + right_eye.x) / 2.0
            mid_y = (left_eye.y + right_eye.y) / 2.0
            face_vec = (nose.x - mid_x, nose.y - mid_y)
            angle = np.degrees(np.arctan2(face_vec[1], face_vec[0])) % 360

            adjusted_angle = (angle + 180) % 360
            if adjusted_angle < 45 or adjusted_angle >= 315:
                orientation = 1  # 左侧
            elif adjusted_angle < 135:
                orientation = 3  # 背面
            elif adjusted_angle < 225:
                orientation = 2  # 右侧
            else:
                if 255 <= adjusted_angle < 285:
                    orientation = 0  # 正面
                elif adjusted_angle < 255:
                    orientation = 1  # 左侧
                else:
                    orientation = 2  # 右侧

            # 若左右耳均有效，则以 (鼻子 - 耳中点) 的向量作为最终方向
            if valid_left_ear and valid_right_ear:
                ear_mid_x = (left_ear.x + right_ear.x) / 2.0
                ear_mid_y = (left_ear.y + right_ear.y) / 2.0
                adj_vec = (nose.x - ear_mid_x, nose.y - ear_mid_y)
                norm = np.sqrt(adj_vec[0] ** 2 + adj_vec[1] ** 2)
                if norm != 0:
                    vec_x, vec_y = adj_vec[0] / norm, adj_vec[1] / norm
                else:
                    vec_x, vec_y = 0.0, 0.0
            else:
                vec_x, vec_y = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
            origin_x, origin_y = int(nose.x), int(nose.y)

        # 情况2：鼻子与仅一只眼有效
        elif valid_nose and (valid_left_eye or valid_right_eye):
            origin_x, origin_y = int(nose.x), int(nose.y)
            if valid_left_eye and not valid_right_eye:
                orientation = 1  # 左侧
                vec_x, vec_y = np.cos(0), np.sin(0)  # (1, 0)
            elif valid_right_eye and not valid_left_eye:
                orientation = 2  # 右侧
                vec_x, vec_y = np.cos(np.deg2rad(180)), np.sin(np.deg2rad(180))  # (-1, 0)

        # 情况3：脸部关键点不可用，但双耳有效 -> 认为是背面
        elif valid_left_ear and valid_right_ear:
            orientation = 3  # 背面
            ear_mid_x = (left_ear.x + right_ear.x) / 2.0
            ear_mid_y = (left_ear.y + right_ear.y) / 2.0
            origin_x, origin_y = int(ear_mid_x), int(ear_mid_y)
            vec_x, vec_y = 0.0, 0.0

        # 情况4：其他情况 -> 保持默认值（未知）
        facial_direction = FacialDirection(
            modulus=modulus,
            vector=(vec_x, vec_y),
            origin=(origin_x, origin_y),
            direction_desc=PoseInsight.ORIENTATION_TEXTS.get(orientation, "Unknown"),
            direction_type=orientation
        )
        return facial_direction

    @staticmethod
    def analyze_poses(poses: List[Union[YoloPose, YoloPoseSorted]]) -> List[Tuple[Posture, FacialDirection]]:
        """
        分析传入的 poses 列表，返回每个 pose 的 Posture 和 FacialDirection 信息，结果格式为：
          [
              (Posture(...), FacialDirection(...)),
              (Posture(...), FacialDirection(...)),
              ...
          ]
        """
        results = []
        for pose in poses:
            posture = PoseInsight._analyze_body_pose(pose)
            facial_direction = PoseInsight._analyze_facial_direction(pose)
            results.append((posture, facial_direction))
        return results

# -------------- 使用示例 --------------
if __name__ == '__main__':

    def test():

        # 假设 poses_list 是一个包含 YoloPose 或 YoloPoseSorted 实例的列表
        poses_list: List[Union[YoloPose, YoloPoseSorted]] = [...]  # 替换为你的数据

        analysis_results = PoseInsight.analyze_poses(poses_list)
        for idx, (posture, facial_direction) in enumerate(analysis_results):
            print(f"Pose {idx + 1}:")
            print("  人体姿态编码：", posture.action)
            print("  面部朝向信息：")
            print("    模长：", facial_direction.modulus)
            print("    方向向量：", facial_direction.vector)
            print("    原点：", facial_direction.origin)
            print("    方向描述：", facial_direction.direction_desc)
            print("    方向编码：", facial_direction.direction_type)
            print("--------------")

    # 运行测试
    test()
