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
    ORIENTATION_TEXTS = {
        0: "Front",
        1: "Left",
        2: "Right",
        3: "Back",
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
        cos_angle = max(min(dot / (norm1 * norm2), 1.0), -1.0)
        angle_rad = math.acos(cos_angle)
        return math.degrees(angle_rad)

    @staticmethod
    def _analyze_body_pose(pose: Union[YoloPose, YoloPoseSorted]) -> Posture:
        """
        根据检测框的长宽比判断人体姿态。
        新的编码规则建议:
            0 - 未知 (Unknown)
            1 - 弯腰 (Bending) - （可选）如果仍需判断弯腰，则需要关键点。
                                如果仅用长宽比，弯腰可能被判断为站立或蹲/坐。
                                这里暂时将其简化，不单独判断弯腰，除非有明确的区分逻辑。
            2 - 坐/下蹲 (Sitting/Squatting)
            3 - 站立 (Standing)
        """
        bbox_width = abs(pose.rx - pose.lx)
        bbox_height = abs(pose.ry - pose.ly)

        if bbox_width == 0 or bbox_height == 0:
            return Posture(action=0)  # 未知，无效检测框
        
        # 检查关键点列表长度是否足够
        if len(pose.pts) < 17: # COCO model has 17 keypoints (0-16)
            # Not enough keypoints to determine ankles, return unknown or a default
            # print(f"Warning: pose.pts has {len(pose.pts)} keypoints, expected at least 17.")
            return Posture(action=0) # Or handle as per your logic

        # COCO 17 keypoints:
        # 15: Left Ankle
        # 16: Right Ankle
        left_ankle = pose.pts[15]  # Corrected index for Left Ankle
        right_ankle = pose.pts[16] # Corrected index for Right Ankle
        if not PoseInsight._is_valid_point(left_ankle) or not PoseInsight._is_valid_point(right_ankle):
            # 如果没有足部关键点，可能是站立或其他姿态
            # 这里可以根据实际情况调整逻辑
            return Posture(action=0)

        aspect_ratio = bbox_height / bbox_width
        # print(aspect_ratio)  # 调试输出，查看长宽比

        # --- 阈值定义 ---
        # 这些阈值需要根据你的具体场景和数据进行调整
        # 通常站立时，高度明显大于宽度
        # 坐或蹲时，高度和宽度相对接近，或者高度略大于/小于宽度
        standing_threshold = 3.0  # 3.0 是一个比较合适的阈值，如果小于3.0，通常被检测人只有一部分身体进入视频中
        # sitting_squatting_threshold_upper = 1.5 # 例如：高/宽 <=1.5
        # sitting_squatting_threshold_lower = 0.7 # 例如：高/宽 >=0.7

        action_code = 0 # 默认为未知

        if aspect_ratio >= standing_threshold:
            action_code = 3  # 站立
        # elif sitting_squatting_threshold_lower <= aspect_ratio < sitting_squatting_threshold_upper:
        #     action_code = 2  # 坐/下蹲
        else:
            action_code = 2 # 坐/下蹲
        
        # TODO
        # 通过bbox比例并不是一个精确的判断方式，之后会通过训练efficent_resnet模型，并增加输出任务来精准的判定
        # 之后这个方法会被舍弃

        return Posture(action=action_code)


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
        if pose.lx > 0 and pose.ly > 0 and pose.rx > pose.lx: # 确保 lx, ly 有效且 rx > lx
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
        if len(pts) < 5: # 至少需要前5个面部关键点
             return FacialDirection(
                modulus=modulus, vector=(0.0, 0.0), origin=(int(pose.lx), int(pose.ly)),
                direction_desc=PoseInsight.ORIENTATION_TEXTS.get(-1, "Unknown"), direction_type=-1
            )

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

        orientation = -1
        vec_x, vec_y = 0.0, 0.0
        origin_x, origin_y = int(pose.lx), int(pose.ly) # 默认原点

        if valid_nose and valid_left_eye and valid_right_eye:
            mid_x = (left_eye.x + right_eye.x) / 2.0
            mid_y = (left_eye.y + right_eye.y) / 2.0
            face_vec = (nose.x - mid_x, nose.y - mid_y) # 从眼中点指向鼻子的向量
            angle = np.degrees(np.arctan2(face_vec[1], face_vec[0])) # 向量与x轴正方向的夹角

            # 角度调整到面部朝向的常规理解 (0度右, 90度下, 180度左, 270度上)
            # 我们需要的是鼻子相对于双眼中心的方向
            # 如果 face_vec[1] (即 nose.y - mid_y) > 0，说明鼻子在双眼下方，朝前
            # 如果 face_vec[0] (即 nose.x - mid_x) > 0，说明鼻子在双眼右侧，脸朝左
            # 如果 face_vec[0] < 0，说明鼻子在双眼左侧，脸朝右

            # 重新定义方向判定逻辑，基于人脸的几何特征
            # eye_dist = np.sqrt((left_eye.x - right_eye.x)**2 + (left_eye.y - right_eye.y)**2)
            # nose_to_left_eye_dist = np.sqrt((nose.x - left_eye.x)**2 + (nose.y - left_eye.y)**2)
            # nose_to_right_eye_dist = np.sqrt((nose.x - right_eye.x)**2 + (nose.y - right_eye.y)**2)

            # 简化：使用之前的角度判断逻辑，但可能需要调整阈值
            # (angle + 180) % 360 的调整可能不直观，直接使用 angle
            # angle: [-180, 180]
            #  ~0 degree: pointing right on image -> face left
            #  ~90 degrees: pointing down on image -> face front
            #  ~-90 degrees: pointing up on image -> face back (less likely)
            #  ~180 or ~-180 degrees: pointing left on image -> face right

            if 45 <= angle <= 135: # 指向图像下方 (大致)
                orientation = 0  # 正面
            elif -45 <= angle < 45: # 指向图像右方
                orientation = 1  # 左侧 (人脸朝左)
            elif -135 <= angle < -45: # 指向图像上方
                orientation = 3  # 背面 (或头顶正对)
            else: # 指向图像左方
                orientation = 2  # 右侧 (人脸朝右)


            if valid_left_ear and valid_right_ear:
                ear_mid_x = (left_ear.x + right_ear.x) / 2.0
                ear_mid_y = (left_ear.y + right_ear.y) / 2.0
                adj_vec = (nose.x - ear_mid_x, nose.y - ear_mid_y)
                norm = np.sqrt(adj_vec[0] ** 2 + adj_vec[1] ** 2)
                if norm != 0:
                    vec_x, vec_y = adj_vec[0] / norm, adj_vec[1] / norm
            else: # 只有双眼和鼻子
                norm_face_vec = np.sqrt(face_vec[0]**2 + face_vec[1]**2)
                if norm_face_vec != 0:
                    vec_x, vec_y = face_vec[0] / norm_face_vec, face_vec[1] / norm_face_vec
                else:
                     vec_x, vec_y = 0.0,0.0
            origin_x, origin_y = int(nose.x), int(nose.y)

        elif valid_nose and (valid_left_eye or valid_right_eye):
            origin_x, origin_y = int(nose.x), int(nose.y)
            if valid_left_eye and not valid_right_eye: # 只能看到左眼和鼻子，说明人脸朝左
                orientation = 1  # 左侧
                vec_x, vec_y = 1.0, 0.0 # 指向图像右侧 (人脸的左边)
            elif valid_right_eye and not valid_left_eye: # 只能看到右眼和鼻子，说明人脸朝右
                orientation = 2  # 右侧
                vec_x, vec_y = -1.0, 0.0 # 指向图像左侧 (人脸的右边)

        elif valid_left_ear and valid_right_ear and not valid_nose and not valid_left_eye and not valid_right_eye:
             # 脸部关键点都不可用，但双耳有效 -> 认为是背面
            orientation = 3  # 背面
            ear_mid_x = (left_ear.x + right_ear.x) / 2.0
            ear_mid_y = (left_ear.y + right_ear.y) / 2.0
            origin_x, origin_y = int(ear_mid_x), int(ear_mid_y)
            vec_x, vec_y = 0.0, -1.0 # 指向图像上方 (通常表示后脑勺)

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
