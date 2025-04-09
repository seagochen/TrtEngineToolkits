from dataclasses import dataclass
from typing import Tuple, Union
import numpy as np
from pyengine.inference.yolo.data_struct import YoloPose, YoloPoseSorted


@dataclass
class FacialVector:
    """
    计算面部向量信息，提供以下接口：
      - module(): 返回模长
      - vector(): 返回单位方向向量 (vec_x, vec_y)
      - original(): 返回原点坐标 (origin_x, origin_y)
      - direction_desc(): 返回方向描述字符串，如 "Front"、"Left"、"Right"、"Back" 或 "Unknown"
      - direction_type(): 返回离散方向编码（0: 正面, 1: 左侧, 2: 右侧, 3: 背面, -1: 未知）

    新的逻辑：
      1. 当鼻子+左右眼均可用时，计算角度（鼻子到眼中点的向量）。
         如果同时左右耳可用，则用 (鼻子 - 两耳中心) 调整方向。
      2. 当仅检测到鼻子+一只眼时，直接逻辑判断为左侧或右侧（向量平行于水平坐标）。
      3. 当脸部关键点不可用，但两只耳可用时，判定为背对，向量长度置 0，原点取两耳中心。
      4. 否则判定为未知，向量长度为 0，原点取检测框的左上角。
    """
    origin_x: int = 0
    origin_y: int = 0
    vec_x: float = 0.0
    vec_y: float = 0.0
    modulus: int = 50
    orientation: int = -1  # 0: 正面, 1: 左侧, 2: 右侧, 3: 背面, -1: 未知

    ORIENTATION_TEXTS = {
        0: "Front",  # 正面
        1: "Left",  # 左侧
        2: "Right",  # 右侧
        3: "Back",  # 背面
        -1: "Unknown"
    }

    def __init__(self, pose: Union[YoloPose, YoloPoseSorted]):
        self._process_pose(pose)

    def _process_pose(self, pose: Union[YoloPose, YoloPoseSorted]):
        # 计算模长（检测框宽度/3，若无效则返回默认值）
        self.modulus = self._compute_modulus(pose)

        # 提取关键点（按照 YOLOPOSE 定义的顺序）
        nose = pose.pts[0]
        right_eye = pose.pts[1]
        left_eye = pose.pts[2]
        right_ear = pose.pts[3]
        left_ear = pose.pts[4]

        valid_nose = self._is_valid_point(nose)
        valid_right_eye = self._is_valid_point(right_eye)
        valid_left_eye = self._is_valid_point(left_eye)
        valid_right_ear = self._is_valid_point(right_ear)
        valid_left_ear = self._is_valid_point(left_ear)

        # 情况1：鼻子 + 双眼均有效 -> 计算角度
        if valid_nose and valid_left_eye and valid_right_eye:
            # 计算眼中点
            mid_x = (left_eye.x + right_eye.x) / 2.0
            mid_y = (left_eye.y + right_eye.y) / 2.0
            # 脸部向量：从眼中点指向鼻子
            face_vec = (nose.x - mid_x, nose.y - mid_y)
            # 计算角度（以水平正方向为参考，单位：度）
            angle = np.degrees(np.arctan2(face_vec[1], face_vec[0])) % 360

            adjusted_angle = (angle + 180) % 360
            if adjusted_angle < 45 or adjusted_angle >= 315:
                self.orientation = 1  # 左侧
            elif adjusted_angle < 135:
                self.orientation = 3  # 背面
            elif adjusted_angle < 225:
                self.orientation = 2  # 右侧
            else:
                # 当 adjusted_angle 在 [225,315) 内时，进一步细分
                if 255 <= adjusted_angle < 285:
                    self.orientation = 0  # 正面（仅当角度非常接近 270° 时认为是正面）
                elif adjusted_angle < 255:
                    self.orientation = 1  # 左侧
                else:  # adjusted_angle >= 285
                    self.orientation = 2  # 右侧

            # 当左右耳均有效时，采用 (鼻子 - 两耳中心) 的向量归一化作为最终方向
            if valid_left_ear and valid_right_ear:
                ear_mid_x = (left_ear.x + right_ear.x) / 2.0
                ear_mid_y = (left_ear.y + right_ear.y) / 2.0
                adj_vec = (nose.x - ear_mid_x, nose.y - ear_mid_y)
                norm = np.sqrt(adj_vec[0] ** 2 + adj_vec[1] ** 2)
                if norm != 0:
                    self.vec_x, self.vec_y = adj_vec[0] / norm, adj_vec[1] / norm
                else:
                    self.vec_x, self.vec_y = 0.0, 0.0
            else:
                # 否则仍使用由眼睛和鼻子计算得到的角度
                self.vec_x, self.vec_y = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
            # 原点取鼻子的位置
            self.origin_x, self.origin_y = int(nose.x), int(nose.y)

        # 情况2：鼻子 + 仅一只眼有效 -> 直接逻辑判断左右
        elif valid_nose and (valid_left_eye or valid_right_eye):
            self.origin_x, self.origin_y = int(nose.x), int(nose.y)
            if valid_left_eye and not valid_right_eye:
                self.orientation = 1  # 左侧
                self.vec_x, self.vec_y = np.cos(np.deg2rad(0)), np.sin(np.deg2rad(0))  # (1, 0)
            elif valid_right_eye and not valid_left_eye:
                self.orientation = 2  # 右侧
                self.vec_x, self.vec_y = np.cos(np.deg2rad(180)), np.sin(np.deg2rad(180))  # (-1, 0)

        # 情况3：脸部关键点不可用，但两只耳均有效 -> 认为是背对
        elif valid_left_ear and valid_right_ear:
            self.orientation = 3  # 背面
            ear_mid_x = (left_ear.x + right_ear.x) / 2.0
            ear_mid_y = (left_ear.y + right_ear.y) / 2.0
            self.origin_x, self.origin_y = int(ear_mid_x), int(ear_mid_y)
            # 背对时向量长度为 0
            self.vec_x, self.vec_y = 0.0, 0.0

        # 情况4：均无法获得有效信息 -> 未知，原点取检测框左上角，向量长度为 0
        else:
            self.orientation = -1
            self.vec_x, self.vec_y = 0.0, 0.0
            self.origin_x, self.origin_y = int(pose.lx), int(pose.ly)

    def module(self) -> int:
        """返回模长"""
        return self.modulus

    def vector(self) -> Tuple[float, float]:
        """返回单位方向向量 (vec_x, vec_y)"""
        return (self.vec_x, self.vec_y)

    def original(self) -> Tuple[int, int]:
        """返回原点坐标 (origin_x, origin_y)"""
        return (self.origin_x, self.origin_y)

    def direction_desc(self) -> str:
        """返回方向描述字符串"""
        return self.ORIENTATION_TEXTS.get(self.orientation, "Unknown")

    def direction_type(self) -> int:
        """返回离散方向编码（0: 正面, 1: 左侧, 2: 右侧, 3: 背面, -1 表示未知）"""
        return self.orientation

    # ---------- 内部辅助函数 ----------
    @staticmethod
    def _is_valid_point(pt) -> bool:
        """判断关键点是否有效：置信度 > 0.2 且坐标均大于0"""
        return pt.conf > 0.2 and pt.x > 0 and pt.y > 0

    @staticmethod
    def _compute_modulus(pose: YoloPose) -> int:
        """
        根据 YoloPose 中检测框的宽度计算模长，
        如果检测框宽度有效，则按宽度计算，否则返回默认值100。
        """
        if pose.lx > 0 and pose.ly > 0:
            return int(np.abs(pose.lx - pose.rx) / 3.0)
        return 100


"""
# 假设 pose 是一个 YoloPose 或 YoloPoseSorted 实例
facial_vec = FacialVector(pose)

module = facial_vec.module()
direction_vect = facial_vec.vector()
original_pt = facial_vec.original()
direction_str = facial_vec.direction_desc()
direction_type = facial_vec.direction_type()

print("模长:", module)
print("方向向量:", direction_vect)
print("原点坐标:", original_pt)
print("方向描述:", direction_str)
print("方向编码:", direction_type)
"""