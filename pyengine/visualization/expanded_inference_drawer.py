import time
from typing import List, Dict, Tuple, Optional, Union, Any

import cv2
import numpy as np

from pyengine.inference.unified_structs.auxiliary_structs import ExpandedSkeleton, FaceDirection
from pyengine.inference.unified_structs.inference_results import Rect, ObjectDetection, Point, Skeleton
from pyengine.utils.logger import logger
from pyengine.visualization.schema_loader import SchemaLoader


class ExpandedInferenceDrawer:
    """
    用于扩展信息绘制
    """

    def __init__(self, schema_file: Optional[str] = None, flashing_frequency_hz: float = 1.0):
        self.schema_loader = SchemaLoader(schema_file)
        self.kpt_color_map = self.schema_loader.kpt_color_map
        self.skeleton_map = self.schema_loader.skeleton_map
        self.bbox_colors = self.schema_loader.bbox_colors  # 这是一个 List[Tuple[int, int, int]]

        # 默认的文本和线条颜色/粗细
        self.text_color = (255, 255, 255)  # White
        self.font_scale = 0.7
        self.thickness = 2
        self.keypoint_radius = 5

        # 缓存 bbox 颜色，确保至少有一个颜色
        if not self.bbox_colors:  # If schema_loader returned an empty list
            logger.warning("GenericInferenceDrawer", "No bbox colors loaded from schema. Using default red.")
            self.bbox_colors = [(0, 0, 255)]  # Default to red if none are loaded

        # Flashing feature additions
        if flashing_frequency_hz <= 0:
            self.flashing_interval = float('inf')  # Effectively disable flashing
        else:
            self.flashing_interval = 1.0 / flashing_frequency_hz
        self._last_flash_time = time.time()
        self._flash_state = False  # False = primary color, True = secondary color

        # Define available flashing color styles (primary_color_for_flash_state_false, primary_color_for_flash_state_true)
        # BGR format
        self._flashing_color_styles: Dict[str, Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = {
            "white_red": ((255, 255, 255), (0, 0, 255)),      # White and Red
            "white_yellow": ((255, 255, 255), (0, 255, 255)), # White and Yellow (BGR)
            "white_green": ((255, 255, 255), (0, 255, 0)),    # White and Green (BGR)
            "white_blue": ((255, 255, 255), (255, 0, 0)),    # White and Blue (BGR)
            "white_cyan": ((255, 255, 255), (255, 255, 0)),    # White and Cyan (BGR)
            "white_magenta": ((255, 255, 255), (255, 0, 255)), # White and Magenta (BGR)
            "white_gray": ((255, 255, 255), (128, 128, 128)), # White and Gray (BGR)
        }

    def _update_flash_state(self):
        """Updates the internal flash state based on time and frequency."""
        current_time = time.time()
        if current_time - self._last_flash_time >= self.flashing_interval:
            self._flash_state = not self._flash_state
            self._last_flash_time = current_time

    def _get_classification_color(self, classification_id: int, flashing: bool = False, flashing_style: str = "white_red") -> Tuple[int, int, int]:
        """
        根据分类ID选择边界框颜色，并支持闪烁功能。
        如果 flashing 为 True，则颜色会根据内部闪烁状态在 self._flash_primary_color 和 self._flash_secondary_color 之间切换。
        Args:
            classification_id: 对象的分类ID。
            flashing: 是否启用闪烁。
            flashing_style: 闪烁模式的颜色风格 (例如 "white_red", "white_yellow")。
        Returns:
            计算出的边界框颜色。
        """
        if flashing:
            # Get the color pair based on the flashing_style
            primary_flash_color, secondary_flash_color = self._flashing_color_styles.get(
                flashing_style,
                self._flashing_color_styles["white_red"] # Fallback to white_red if style not found
            )
            self._update_flash_state()
            return secondary_flash_color if self._flash_state else primary_flash_color
        else:
            color_at_index = classification_id % len(self.bbox_colors)  # 确保索引在范围内
            return self.bbox_colors[color_at_index]

    @staticmethod
    def _scale_coordinates(coords: Union[Rect, Point, Tuple[float, float]],
                           original_shape: Tuple[int, int],
                           pipeline_input_shape: Tuple[int, int] = (640, 640)) -> Any:
        """
        根据图像从 pipeline 输入尺寸缩放到原始尺寸的比例调整坐标。
        Args:
            coords: Rect, Point 或 (x, y) 元组。
            original_shape: 原始图像的 (height, width)。
            pipeline_input_shape: pipeline 内部处理的图像尺寸 (height, width)。
        Returns:
            缩放后的 Rect, Point 或 (int, int) 元组。
        """
        original_h, original_w = original_shape
        pipeline_h, pipeline_w = pipeline_input_shape
        scale_x = original_w / pipeline_w
        scale_y = original_h / pipeline_h

        if isinstance(coords, Rect):
            return Rect(
                x1=coords.x1 * scale_x, y1=coords.y1 * scale_y,
                x2=coords.x2 * scale_x, y2=coords.y2 * scale_y
            )
        elif isinstance(coords, Point):
            return Point(
                x=int(coords.x * scale_x), y=int(coords.y * scale_y),
                confidence=coords.confidence
            )
        elif isinstance(coords, tuple) and len(coords) == 2:
            return int(coords[0] * scale_x), int(coords[1] * scale_y)
        else:
            return coords  # Return as is if not a recognized type

    def draw_object_detection(self,
                              image: np.ndarray,
                              detection: ObjectDetection,
                              original_image_shape: Tuple[int, int],
                              enable_track_id: bool = True,
                              label_names: Optional[List[str]] = None,
                              flashing_bbox: bool = False,
                              flashing_style: str = "white_red") -> np.ndarray: # Added flashing_style
        """
        在图像上绘制一个 ObjectDetection 对象。
        Args:
            image: 要绘制的 NumPy 图像数组 (BGR)。
            detection: ObjectDetection 实例。
            original_image_shape: 原始图像的 (height, width)，用于坐标缩放。
            enable_track_id: 是否在标签中显示 track_id。
            label_names: 分类标签名称列表 (例如 ['person', 'cat'])。
                         如果提供，将显示分类名称，否则显示分类 ID。
            flashing_bbox: 是否让边界框闪烁。
            flashing_style: 闪烁模式的颜色风格 (例如 "white_red", "white_yellow")。
        Returns:
            绘制后的图像副本。
        """
        display_image = image.copy()

        # 缩放边界框坐标
        scaled_rect = self._scale_coordinates(detection.rect, original_image_shape)
        x1, y1, x2, y2 = map(int, [scaled_rect.x1, scaled_rect.y1, scaled_rect.x2, scaled_rect.y2])

        # 获取边界框颜色 - 现在根据 classification 来获取，并考虑闪烁
        bbox_color = self._get_classification_color(detection.classification, flashing=flashing_bbox, flashing_style=flashing_style) # Pass style

        # 构建标签文本
        if label_names and 0 <= detection.classification < len(label_names):
            class_label_display = label_names[detection.classification]
        else:
            class_label_display = "Class {}".format(detection.classification)

        label_text = ""
        if enable_track_id and detection.track_id > 0:
            label_text = f"#{detection.track_id} "
        label_text += f"{class_label_display} {detection.confidence:.2f}"

        # 绘制边界框
        cv2.rectangle(display_image, (x1, y1), (x2, y2), bbox_color, self.thickness)

        # 绘制标签文本 (在边界框上方)
        text_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.thickness)
        text_x = x1
        text_y = y1 - 10
        if text_y < 0:
            text_y = y1 + text_size[1] + 10

        cv2.rectangle(display_image, (text_x, text_y - text_size[1] - 5),
                      (text_x + text_size[0], text_y + 5), bbox_color, -1)
        cv2.putText(display_image, label_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.text_color, self.thickness)

        return display_image

    def draw_skeleton(self,
                      image: np.ndarray,
                      skeleton: Skeleton,
                      original_image_shape: Tuple[int, int],
                      enable_track_id: bool = True,
                      label_names: Optional[List[str]] = None,
                      enable_pts_names: bool = False,
                      enable_skeleton: bool = True,
                      flashing_bbox: bool = False,
                      flashing_style: str = "white_red") -> np.ndarray: # Added flashing_style
        """
        在图像上绘制一个 Skeleton 对象。
        Args:
            image: 要绘制的 NumPy 图像数组 (BGR)。
            skeleton: Skeleton 实例。
            original_image_shape: 原始图像的 (height, width)，用于坐标缩放。
            enable_track_id: 是否在标签中显示 track_id。
            label_names: 分类标签名称列表 (例如 ['girl', 'boy'])。
                         如果提供，将显示分类名称，否则统一显示 'person'。
            enable_pts_names: 如果为 True，将在关键点附近添加名字。
            enable_skeleton: 如果为 True，绘制关键点和骨骼连线。
            flashing_bbox: 是否让边界框闪烁。
            flashing_style: 闪烁模式的颜色风格 (例如 "white_red", "white_yellow")。
        Returns:
            绘制后的图像副本。
        """
        display_image = image.copy()

        # 绘制边界框 (继承自 ObjectDetection)
        effective_label_names = ['person'] if not label_names else label_names
        display_image = self.draw_object_detection(
            display_image,
            skeleton,
            original_image_shape,
            enable_track_id=enable_track_id,
            label_names=effective_label_names,
            flashing_bbox=flashing_bbox,
            flashing_style=flashing_style # Pass style
        )

        if not enable_skeleton:
            return display_image

        # 缩放所有关键点
        scaled_points = [
            self._scale_coordinates(p, original_image_shape, pipeline_input_shape=(640, 640))
            for p in skeleton.points
        ]

        # 将关键点转换为字典，以便通过 ID 快速查找
        points_by_id: Dict[int, Point] = {i: p for i, p in enumerate(scaled_points)}

        # 绘制骨骼连线
        for link in self.skeleton_map:
            srt_kpt = points_by_id.get(link.srt_kpt_id)
            dst_kpt = points_by_id.get(link.dst_kpt_id)

            if srt_kpt and dst_kpt and srt_kpt.confidence > 0.1 and dst_kpt.confidence > 0.1:
                cv2.line(display_image,
                         (int(srt_kpt.x), int(srt_kpt.y)),
                         (int(dst_kpt.x), int(dst_kpt.y)),
                         link.color, self.thickness)

        # 绘制关键点
        for kpt_id, kp in points_by_id.items():
            if kp.confidence > 0.1:
                center = (int(kp.x), int(kp.y))
                color_info = self.kpt_color_map.get(kpt_id)
                kpt_color = color_info.color if color_info else (255, 255, 255)

                cv2.circle(display_image, center, self.keypoint_radius, kpt_color, -1)

                if enable_pts_names and color_info and color_info.name:
                    kpt_name_text = color_info.name
                    cv2.putText(display_image, kpt_name_text,
                                (center[0] + self.keypoint_radius + 2, center[1] + self.keypoint_radius + 2),
                                cv2.FONT_HERSHEY_SIMPLEX, self.font_scale * 0.6, self.text_color, 1)

        return display_image

    def draw_object_detections_batch(self,
                                     image: np.ndarray,
                                     detections: List[ObjectDetection],
                                     original_image_shape: Tuple[int, int],
                                     enable_track_id: bool = True,
                                     label_names: Optional[List[str]] = None,
                                     flashing_bbox: bool = False,
                                     flashing_style: str = "white_red") -> np.ndarray: # Added flashing_style
        """
        在图像上批量绘制 ObjectDetection 对象。
        Args:
            image: 要绘制的 NumPy 图像数组 (BGR)。
            detections: ObjectDetection 实例的列表。
            original_image_shape: 原始图像的 (height, width)，用于坐标缩放。
            enable_track_id: 是否在标签中显示 track_id。
            label_names: 分类标签名称列表。
            flashing_bbox: 是否让边界框闪烁。
            flashing_style: 闪烁模式的颜色风格 (例如 "white_red", "white_yellow")。
        Returns:
            绘制后的图像副本。
        """
        display_image = image.copy()
        for det in detections:
            display_image = self.draw_object_detection(
                display_image, det, original_image_shape, enable_track_id, label_names, flashing_bbox, flashing_style # Pass style
            )
        return display_image

    def draw_skeletons_batch(self,
                             image: np.ndarray,
                             skeletons: List[Skeleton],
                             original_image_shape: Tuple[int, int],
                             enable_track_id: bool = True,
                             label_names: Optional[List[str]] = None,
                             enable_pts_names: bool = False,
                             enable_skeleton: bool = True,
                             flashing_bbox: bool = False,
                             flashing_style: str = "white_red") -> np.ndarray: # Added flashing_style
        """
        在图像上批量绘制 Skeleton 对象。
        Args:
            image: 要绘制的 NumPy 图像数组 (BGR)。
            skeletons: Skeleton 实例的列表。
            original_image_shape: 原始图像的 (height, width)，用于坐标缩放。
            enable_track_id: 是否在标签中显示 track_id。
            label_names: 分类标签名称列表。
            enable_pts_names: 如果为 True，将在关键点附近添加名字。
            enable_skeleton: 如果为 True，绘制关键点和骨骼连线。
            flashing_bbox: 是否让边界框闪烁。
            flashing_style: 闪烁模式的颜色风格 (例如 "white_red", "white_yellow")。
        Returns:
            绘制后的图像副本。
        """
        display_image = image.copy()
        for skel in skeletons:
            display_image = self.draw_skeleton(
                display_image, skel, original_image_shape, enable_track_id,
                label_names, enable_pts_names, enable_skeleton, flashing_bbox, flashing_style # Pass style
            )
        return display_image

    def draw_facial_orientation_vector(self,
                                       image: np.ndarray,
                                       expanded_skeleton: ExpandedSkeleton,
                                       original_image_shape: Tuple[int, int],
                                       color: Tuple[int, int, int] = (0, 255, 0),  # Default Green
                                       thickness: int = 2,
                                       show_dir_name: bool = True,
                                       font_scale: float = 0.5) -> np.ndarray:
        """
        根据 ExpandedSkeleton 的面部朝向信息绘制视线方向向量。
        Args:
            image: 要绘制的 NumPy 图像数组 (BGR)。
            expanded_skeleton: ExpandedSkeleton 实例，包含面部朝向信息。
            original_image_shape: 原始图像的 (height, width)，用于坐标缩放。
            color: 箭头的颜色。
            thickness: 箭头的粗细。
            show_dir_name: 是否显示方向名称。
            font_scale: 方向名称的字体大小。
        Returns:
            绘制后的图像副本。
        """
        display_image = image.copy()

        # 获取缩放后的关键点
        scaled_points = [
            self._scale_coordinates(p, original_image_shape, pipeline_input_shape=(640, 640))
            for p in expanded_skeleton.points
        ]
        points_by_id: Dict[int, Point] = {i: p for i, p in enumerate(scaled_points)}

        # 定义关键点ID (假设根据 COCO 或类似的骨骼模型)
        NOSE_ID = 0
        LEFT_EAR_ID = 4 # 左耳ID (assuming a consistent schema like COCO)
        RIGHT_EAR_ID = 3 # 右耳ID (assuming a consistent schema like COCO)

        start_point = None
        end_point = None
        direction_text = expanded_skeleton.direction_type.name

        # 根据 direction_type 决定绘制逻辑
        if expanded_skeleton.direction_type == FaceDirection.Front:
            # 当 FaceDirection == Front 时，从鼻尖绘制向量
            nose_kpt = points_by_id.get(NOSE_ID)
            if nose_kpt and nose_kpt.confidence > 0.1:
                start_point = (int(nose_kpt.x), int(nose_kpt.y))
                # 使用 direction_vector 和 direction_modulus
                vec_x, vec_y = expanded_skeleton.direction_vector
                modulus = expanded_skeleton.direction_modulus

                if modulus > 0 and (vec_x != 0 or vec_y != 0):
                    # 根据向量和模长计算终点
                    # 这里的 direction_vector 假设已经是单位向量或其方向正确
                    # 如果 direction_vector 是归一化的，直接乘以 modulus
                    end_point = (int(nose_kpt.x + vec_x * modulus), int(nose_kpt.y + vec_y * modulus))
                else:
                    # 如果模长为0或向量为0，画一个小点表示
                    cv2.circle(display_image, start_point, thickness + 1, color, -1)  # Filled circle

        elif expanded_skeleton.direction_type == FaceDirection.Left:
            # 当 FaceDirection == Left 时，从左耳到鼻子延长线绘制
            left_ear_kpt = points_by_id.get(LEFT_EAR_ID)
            nose_kpt = points_by_id.get(NOSE_ID)
            if left_ear_kpt and nose_kpt and left_ear_kpt.confidence > 0.1 and nose_kpt.confidence > 0.1:
                # 计算从左耳到鼻子的向量
                vec_x = nose_kpt.x - left_ear_kpt.x
                vec_y = nose_kpt.y - left_ear_kpt.y

                # 延长线，例如延长两倍长度
                extension_factor = 2.0
                start_point = (int(left_ear_kpt.x), int(left_ear_kpt.y))
                end_point = (int(left_ear_kpt.x + vec_x * extension_factor),
                             int(left_ear_kpt.y + vec_y * extension_factor))

        elif expanded_skeleton.direction_type == FaceDirection.Right:
            # 当 FaceDirection == Right 时，从右耳到鼻子延长线绘制
            right_ear_kpt = points_by_id.get(RIGHT_EAR_ID)
            nose_kpt = points_by_id.get(NOSE_ID)
            if right_ear_kpt and nose_kpt and right_ear_kpt.confidence > 0.1 and nose_kpt.confidence > 0.1:
                # 计算从右耳到鼻子的向量
                vec_x = nose_kpt.x - right_ear_kpt.x
                vec_y = nose_kpt.y - right_ear_kpt.y

                # 延长线，例如延长两倍长度
                extension_factor = 2.0
                start_point = (int(right_ear_kpt.x), int(right_ear_kpt.y))
                end_point = (int(right_ear_kpt.x + vec_x * extension_factor),
                             int(right_ear_kpt.y + vec_y * extension_factor))

        # 绘制箭头
        if start_point and end_point:
            cv2.arrowedLine(display_image, start_point, end_point, color, thickness, line_type=cv2.LINE_AA)

        # 显示方向名称
        if show_dir_name and start_point:
            text_pos = (start_point[0] + 5, start_point[1] - 5)
            cv2.putText(display_image, direction_text, text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

        return display_image