from typing import List, Union, Tuple

import cv2
import numpy as np

from pyengine.inference.yolo.data_struct import YoloPose, Yolo, YoloPoseSorted, YoloSorted
from pyengine.inference.yolo.pose_insight import FacialDirection
from pyengine.inference.yolo.schema_loader import SchemaLoader


class InferenceDrawer:
    """Drawing utility class for visualizing skeletons, keypoints, and bounding boxes on images."""

    def __init__(self, schema_loader: SchemaLoader, flashing_countdown: int = 5):
        self.kpt_color_map = schema_loader.kpt_color_map
        self.skeleton_map = schema_loader.skeleton_map
        self.bbox_colors = schema_loader.bbox_colors
        self.swap_flashing_flag = 0
        self.flashing_countdown = flashing_countdown
        self.current_flashing_countdown = flashing_countdown


    @staticmethod
    def _draw_color_bbox(image: np.ndarray,
                         text: str,
                         text_color: Tuple[int, int, int],
                         bbox_coords: Tuple[int, int, int, int],
                         bbox_color: Tuple[int, int, int]):
        """
        Draw bounding box with background text on the provided image copy.
        """

        # 绘制包裹物体的框
        cv2.rectangle(image,
                      (bbox_coords[0], bbox_coords[1]),
                      (bbox_coords[2], bbox_coords[3]),
                      bbox_color,
                      1)

        # 绘制文字
        if text:
            # 计算文字的大小
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

            # 绘制包含文字的框
            cv2.rectangle(image,
                          (bbox_coords[0], bbox_coords[1] - text_height - 5),
                          (bbox_coords[0] + text_width, bbox_coords[1]),
                          bbox_color,
                          -1)

            # 绘制文字
            cv2.putText(image,
                        text,
                        (bbox_coords[0], bbox_coords[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        text_color,
                        1)


    def _update_flashing_status(self):
        """Update flashing status."""
        if self.current_flashing_countdown == 0:
            self.swap_flashing_flag = 1 - self.swap_flashing_flag
            self.current_flashing_countdown = self.flashing_countdown
        self.current_flashing_countdown -= 1


    def _get_bbox_color(self, bbox_style: str, oid: int = None):
        """
        Retrieve bounding box color based on style.
        当 bbox_style 为闪烁类型时，返回对应闪烁颜色；
        当 bbox_style 为 "chromatic" 时，根据传入的 oid 选择 schema 中对应的颜色；
        否则返回预设的普通颜色。
        """
        flash_colors = {
            "flash_red_white": [(0, 0, 255), (255, 255, 255)],          # 红白闪烁
            "flash_green_white": [(0, 255, 0), (255, 255, 255)],        # 绿白闪烁
            "flash_yellow_white": [(0, 255, 255), (255, 255, 255)],     # 黄白闪烁
            "flash_blue_white": [(255, 0, 0), (255, 255, 255)]          # 蓝白闪烁
        }
        if bbox_style in flash_colors:
            return flash_colors[bbox_style][self.swap_flashing_flag]  # 使用闪烁颜色

        if bbox_style == "chromatic":
            if oid is None:
                return [255, 255, 255]
            return self.bbox_colors[oid % len(self.bbox_colors)]

        normal_colors = {
            "white": (255, 255, 255),
            "red": (0, 0, 255),
            "green": (0, 255, 0),
            "blue": (255, 0, 0),
            "yellow": (255, 255, 0),
            "gray": (128, 128, 128),
        }
        return normal_colors.get(bbox_style, (255, 255, 255))  # 默认返回白色


    def draw_skeletons(self,
                       frame: np.ndarray,
                       results: List[Union[YoloPose, YoloPoseSorted]],
                       conf_threshold: float = 0.5,
                       bbox_style: str = "blue",
                       show_skeletons: bool = True,
                       show_pts: bool = True,
                       show_pts_name: bool = True) -> np.ndarray:
        """
        Draw skeletons on an image copy.
        """
        frame_copy = frame.copy()
        self._update_flashing_status()

        for pose in results:
            # 绘制关键点
            if show_pts and show_skeletons:
                for idx, kpt in enumerate(pose.pts):
                    if kpt.conf > conf_threshold and idx in self.kpt_color_map:
                        kp = self.kpt_color_map[idx]
                        cv2.circle(frame_copy, (int(kpt.x), int(kpt.y)), 3, kp.color, -1)
                        if show_pts_name:
                            cv2.putText(frame_copy, kp.name, (int(kpt.x), int(kpt.y - 5)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, kp.color, 1)

            # 绘制骨架连线
            if show_skeletons:
                for bone in self.skeleton_map:
                    srt_kp = pose.pts[bone.srt_kpt_id]
                    dst_kp = pose.pts[bone.dst_kpt_id]
                    if all([srt_kp.conf > conf_threshold, dst_kp.conf > conf_threshold,
                            srt_kp.x > 0, srt_kp.y > 0, dst_kp.x > 0, dst_kp.y > 0]):
                        cv2.line(frame_copy, (int(srt_kp.x), int(srt_kp.y)),
                                 (int(dst_kp.x), int(dst_kp.y)), bone.color, 1)

            # 绘制目标 bbox（在 bbox 上方显示置信度信息）
            if bbox_style == "chromatic":
                bbox_color = self._get_bbox_color(bbox_style, pose.oid)
                text = f"ID {pose.oid}: {pose.conf:.2f}"
            else:
                bbox_color = self._get_bbox_color(bbox_style)
                text = f"{pose.conf:.2f}"
            self._draw_color_bbox(frame_copy,
                                  text,
                                  (255, 255, 255),
                                  (pose.lx, pose.ly, pose.rx, pose.ry),
                                  bbox_color)

        return frame_copy


    def draw_objects(self,
                     frame: np.ndarray,
                     results: List[Union[Yolo, YoloSorted]],
                     labels: List[Union[str, int]] = None,
                     bbox_style: str = "blue",
                     show_default=False) -> np.ndarray:
        """
        Draw object bounding boxes on an image copy.
        """
        frame_copy = frame.copy()
        self._update_flashing_status()

        for obj in results:

            if bbox_style == "chromatic":
                oid = obj.oid if hasattr(obj, "oid") else None
                bbox_color = self._get_bbox_color(bbox_style, oid)
            else:
                bbox_color = self._get_bbox_color(bbox_style)

            # 获取id值
            id_value = obj.oid if hasattr(obj, "oid") else obj.cls

            # 决定文字内容：show_default 与 labels 互斥
            if show_default:
                text = f"ID {id_value}: {obj.conf:.2f}"
            else:
                if labels and isinstance(labels, list):
                    label_len = len(labels)
                    text = f"{labels[obj.cls % label_len]}: {obj.conf:.2f} - {id_value}"
                elif labels and isinstance(labels, str):
                    text = f"{labels}"
                else:
                    text = None

            # 绘制bbox以及文字（如果有）
            self._draw_color_bbox(frame_copy,
                                  text,
                                  (255, 255, 255),
                                  (obj.lx, obj.ly, obj.rx, obj.ry),
                                  bbox_color)

        return frame_copy


    def draw_facial_orientation_vectors(self,
                                        image: np.ndarray,
                                        facial_vectors: List[FacialDirection],
                                        bbox_style: str = "blue",
                                        thickness=2,
                                        show_dir_name: bool = False) -> np.ndarray:
        """
        Draw facial orientation vectors on an image copy.
        """
        image_copy = image.copy()
        default_color = self._get_bbox_color(bbox_style)

        for vector in facial_vectors:
            op_x, op_y  = vector.origin             # 起点坐标
            module = vector.modulus                 # 模长
            vec_x, vec_y = vector.vector            # 方向向量
            orientation = vector.direction_desc     # 朝向类型文字

            # 绘制面部朝向箭头
            end_point = (int(op_x + vec_x * module), int(op_y + vec_y * module)) # 终点坐标
            start_point = (int(op_x), int(op_y))
            cv2.arrowedLine(image_copy, start_point, end_point, default_color, thickness)

            # 绘制朝向类型文字
            if show_dir_name:
                orientation_text = f"{orientation}"
                cv2.putText(image_copy, orientation_text, (start_point[0] + 10, start_point[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, default_color, 1, cv2.LINE_AA)

        return image_copy


    def draw_pose_estimation(self,
                             image: np.ndarray,
                             results: List[Union[YoloPose, YoloPoseSorted]],
                             actions: List[int],
                             bbox_style: str = "blue",
                             font_scale=0.5,
                             thickness: int = 1) -> np.ndarray:
        """
        Draw pose estimation action text on an image copy.
        将动作信息绘制到每个目标 bbox 框内的左上角。

        参数 actions 为一组动作代码，定义如下：
            0 - 未知
            1 - 弯腰
            2 - 坐
            3 - 下蹲
            4 - 站立

        bbox_style 用于设置文字背景的底色（通过 _get_bbox_color 获取颜色），文字颜色固定为白色。
        """
        image_copy = image.copy()

        if len(actions) != len(results):
            raise ValueError("动作信息的数量必须与目标数量一致")

        action_map = {
            0: "Unkown",
            1: "Bending",
            2: "Sitting",
            3: "Squatting",
            4: "Standing"
        }

        for obj, act in zip(results, actions):
            text = action_map.get(act, "Unkown")
            # 以目标 bbox 的左上角作为文本绘制起点
            x, y = obj.lx, obj.ly
            font = cv2.FONT_HERSHEY_SIMPLEX
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            margin = 2
            # 计算文本背景区域（在 bbox 内绘制）
            rect_tl = (x, y)
            rect_br = (x + text_width + 2 * margin, y + text_height + baseline + 2 * margin)
            oid = getattr(obj, "oid", None)
            bgcolor = self._get_bbox_color(bbox_style, oid)
            cv2.rectangle(image_copy, rect_tl, rect_br, bgcolor, thickness=-1)
            cv2.putText(image_copy, text, (x + margin, y + text_height + margin),
                        font, font_scale, (255, 255, 255), thickness)

        return image_copy