import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Union

from pyengine.utils.logger import logger
from pyengine.inference.unified_structs.inference_results import Rect, ObjectDetection, Point, Skeleton
from pyengine.visualization.schema_loader import SchemaLoader


class GenericInferenceDrawer:
    """
    用于绘制 ObjectDetection 和 Skeleton 对象的绘图类。
    利用 SchemaLoader 加载颜色和骨骼连接配置。
    """

    def __init__(self, schema_file: Optional[str] = None):
        self.schema_loader = SchemaLoader(schema_file)
        self.kpt_color_map = self.schema_loader.kpt_color_map
        self.skeleton_map = self.schema_loader.skeleton_map
        self.bbox_colors = self.schema_loader.bbox_colors

        # 默认的文本和线条颜色/粗细
        self.text_color = (255, 255, 255)  # White
        self.font_scale = 0.7
        self.thickness = 2
        self.keypoint_radius = 5

        # 缓存 bbox 颜色，按索引选择
        self._num_bbox_colors = len(self.bbox_colors)
        if self._num_bbox_colors == 0:
            logger.warning("ObjectDrawer", "No bbox colors loaded from schema. Using default red.")
            self.bbox_colors = [(0, 0, 255)]  # Default to red if none are loaded
            self._num_bbox_colors = 1

    def _get_bbox_color(self, track_id: int) -> Tuple[int, int, int]:
        """根据 track_id 选择一个循环的边界框颜色。"""
        if track_id is None or track_id < 0:  # For detections before tracking, or untracked ones
            return 0, 255, 255  # Default cyan for untracked/unassigned
        return self.bbox_colors[track_id % self._num_bbox_colors]

    @staticmethod # 修正：移除了 'self' 参数
    def _scale_coordinates(coords: Union[Rect, Point, Tuple[float, float]],
                           original_shape: Tuple[int, int],
                           pipeline_input_shape: Tuple[int, int] = (640, 640)) -> Rect | Point | tuple[int, int] | \
                                                                                  tuple[float, float]:
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
                              label_names: Optional[List[str]] = None) -> np.ndarray:
        """
        在图像上绘制一个 ObjectDetection 对象。
        Args:
            image: 要绘制的 NumPy 图像数组 (BGR)。
            detection: ObjectDetection 实例。
            original_image_shape: 原始图像的 (height, width)，用于坐标缩放。
            enable_track_id: 是否在标签中显示 track_id。
            label_names: 分类标签名称列表 (例如 ['person', 'cat'])。
                         如果提供，将显示分类名称，否则显示分类 ID。
        Returns:
            绘制后的图像副本。
        """
        display_image = image.copy()

        # 缩放边界框坐标
        # 修正：调用静态方法时，不需要传入 self
        scaled_rect = self._scale_coordinates(detection.rect, original_image_shape)
        x1, y1, x2, y2 = map(int, [scaled_rect.x1, scaled_rect.y1, scaled_rect.x2, scaled_rect.y2])

        # 获取边界框颜色
        bbox_color = self._get_bbox_color(detection.track_id)

        # 构建标签文本
        class_label = str(detection.classification)
        if label_names and 0 <= detection.classification < len(label_names):
            class_label = label_names[detection.classification]

        label_text = ""
        if enable_track_id:
            label_text = f"ID: {detection.track_id} - "
        label_text += f"Class: {class_label} Score: {detection.confidence:.2f}"

        # 绘制边界框
        cv2.rectangle(display_image, (x1, y1), (x2, y2), bbox_color, self.thickness)

        # 绘制标签文本 (在边界框上方)
        text_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.thickness)
        text_x = x1
        text_y = y1 - 10
        if text_y < 0:  # Ensure text is visible
            text_y = y1 + text_size[1] + 10

        cv2.rectangle(display_image, (text_x, text_y - text_size[1] - 5),
                      (text_x + text_size[0], text_y + 5), bbox_color, -1)  # Background for text
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
                      enable_skeleton: bool = True) -> np.ndarray:
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
        Returns:
            绘制后的图像副本。
        """
        display_image = image.copy()

        # 绘制边界框 (继承自 ObjectDetection)
        # 修正：调用静态方法时，不需要传入 self
        display_image = self.draw_object_detection(
            display_image,
            skeleton,  # Skeleton 也是 ObjectDetection
            original_image_shape,
            enable_track_id=enable_track_id,
            # 对于 Skeleton，分类标签优先使用提供的 label_names，否则统一为 'person'
            label_names=['person'] if not label_names else label_names
        )

        if not enable_skeleton:
            return display_image

        # 缩放所有关键点
        # 修正：调用静态方法时，不需要传入 self
        scaled_points = [
            self._scale_coordinates(p, original_image_shape, pipeline_input_shape=(640, 640))
            for p in skeleton.points
        ]

        # 将关键点转换为字典，以便通过 ID 快速查找
        # 注意: 关键点的 ID 是从 0 到 16，对应 JSON schema 中的 key
        points_by_id: Dict[int, Point] = {i: p for i, p in enumerate(scaled_points)}

        # 绘制骨骼连线
        for link in self.skeleton_map:
            srt_kpt = points_by_id.get(link.srt_kpt_id)
            dst_kpt = points_by_id.get(link.dst_kpt_id)

            if srt_kpt and dst_kpt and srt_kpt.confidence > 0.1 and dst_kpt.confidence > 0.1:  # 仅绘制置信度高的连线
                cv2.line(display_image,
                         (int(srt_kpt.x), int(srt_kpt.y)),
                         (int(dst_kpt.x), int(dst_kpt.y)),
                         link.color, self.thickness)

        # 绘制关键点
        for kpt_id, kp in points_by_id.items():
            if kp.confidence > 0.1:  # 仅绘制置信度高的关键点
                center = (int(kp.x), int(kp.y))
                color_info = self.kpt_color_map.get(kpt_id)
                kpt_color = color_info.color if color_info else (255, 255, 255)  # Default to white if not found

                cv2.circle(display_image, center, self.keypoint_radius, kpt_color, -1)  # -1 means filled circle

                # 绘制关键点名称
                if enable_pts_names and color_info and color_info.name:
                    kpt_name_text = color_info.name
                    cv2.putText(display_image, kpt_name_text,
                                (center[0] + self.keypoint_radius + 2, center[1] + self.keypoint_radius + 2),  # Offset
                                cv2.FONT_HERSHEY_SIMPLEX, self.font_scale * 0.6, self.text_color, 1)  # Smaller font

        return display_image

    def draw_object_detections_batch(self,
                                     image: np.ndarray,
                                     detections: List[ObjectDetection],
                                     original_image_shape: Tuple[int, int],
                                     enable_track_id: bool = True,
                                     label_names: Optional[List[str]] = None) -> np.ndarray:
        """
        在图像上批量绘制 ObjectDetection 对象。
        Args:
            image: 要绘制的 NumPy 图像数组 (BGR)。
            detections: ObjectDetection 实例的列表。
            original_image_shape: 原始图像的 (height, width)，用于坐标缩放。
            enable_track_id: 是否在标签中显示 track_id。
            label_names: 分类标签名称列表。
        Returns:
            绘制后的图像副本。
        """
        display_image = image.copy()
        for det in detections:
            display_image = self.draw_object_detection(
                display_image, det, original_image_shape, enable_track_id, label_names
            )
        return display_image

    def draw_skeletons_batch(self,
                             image: np.ndarray,
                             skeletons: List[Skeleton],
                             original_image_shape: Tuple[int, int],
                             enable_track_id: bool = True,
                             label_names: Optional[List[str]] = None,
                             enable_pts_names: bool = False,
                             enable_skeleton: bool = True) -> np.ndarray:
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
        Returns:
            绘制后的图像副本。
        """
        display_image = image.copy()
        for skel in skeletons:
            display_image = self.draw_skeleton(
                display_image, skel, original_image_shape, enable_track_id,
                label_names, enable_pts_names, enable_skeleton
            )
        return display_image