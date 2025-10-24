# inference_drawer.py
from typing import List, Dict, Tuple

import cv2
import numpy as np

from pyengine.font import text_painter
from pyengine.inference.unified_structs.auxiliary_structs import ExpandedSkeleton, FaceDirection
from pyengine.inference.unified_structs.inference_results import Skeleton
from pyengine.visualization.scheme_loader import SchemeLoader


class InferenceDrawer:
    """
    统一、模块化、可配置的推理结果绘制类。

    设计原则：
    -   只负责在原图坐标系上绘制，不做任何缩放。
    -   传入的 `skeletons` 坐标必须与 `image` 的坐标系一致。
    -   如需缩放显示，请在绘制后使用 cv2.resize() 手动缩放。
    -   这样设计更清晰，避免了隐式坐标转换导致的错位问题。

    用法示例:

    # 初始化
    drawer = InferenceDrawer(
        scheme_loader=SchemeLoader(SCHEME_CONFIG)
    )

    # 绘制（坐标系与 image 一致）
    frame_with_results = drawer.draw_inference(frame, skeletons, ...)

    # 如需缩放，手动控制
    display_frame = cv2.resize(frame_with_results, (1024, 600))
    cv2.imshow("Window", display_frame)

    """

    def __init__(self,
                 scheme_loader: SchemeLoader,
                 *,
                 bbox_confidence_threshold: float = 0.5,
                 kpt_confidence_threshold: float = 0.5):

        self.bbox_conf_thresh = bbox_confidence_threshold
        self.kpt_conf_thresh = kpt_confidence_threshold
        self.schema = scheme_loader
        self.blink_counter = 0

    # ---------- 坐标映射工具（简化版：不做缩放）----------
    def _map_point(self, x: float, y: float) -> Tuple[int, int]:
        """
        将浮点坐标转换为整数像素坐标（不做缩放）。
        """
        return int(round(x)), int(round(y))

    def _map_rect(self, rect) -> Tuple[int, int, int, int]:
        """
        将矩形坐标转换为整数像素坐标（不做缩放）。
        """
        return int(round(rect.x1)), int(round(rect.y1)), int(round(rect.x2)), int(round(rect.y2))

    # ---------- 具体绘制 ----------
    def _draw_bbox(self, image: np.ndarray, skeleton: Skeleton, bbox_color: Tuple[int, int, int] = None):
        # 使用新的映射函数（不再需要 image 参数）
        x1, y1, x2, y2 = self._map_rect(skeleton.rect)

        if bbox_color is None:
            bbox_colors = self.schema.bbox_colors
            bbox_color = bbox_colors[skeleton.classification % len(bbox_colors)]

        cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, 2)

    def _draw_keypoints(self, image: np.ndarray, skeleton: Skeleton):
        for i, point in enumerate(skeleton.points):
            if point.confidence > self.kpt_conf_thresh:
                # 使用新的映射函数
                kx, ky = self._map_point(point.x, point.y)
                kpt_schema = self.schema.kpt_color_map.get(i)
                if kpt_schema:
                    color = kpt_schema.color
                    cv2.circle(image, (kx, ky), 5, color, -1)

    def _draw_skeleton_links(self, image: np.ndarray, skeleton: Skeleton):
        for link_schema in self.schema.skeleton_map:
            p1_idx, p2_idx = link_schema.srt_kpt_id, link_schema.dst_kpt_id
            if p1_idx < len(skeleton.points) and p2_idx < len(skeleton.points):
                p1 = skeleton.points[p1_idx]
                p2 = skeleton.points[p2_idx]
                if p1.confidence > self.kpt_conf_thresh and p2.confidence > self.kpt_conf_thresh:
                    # 使用新的映射函数
                    x1, y1 = self._map_point(p1.x, p1.y)
                    x2, y2 = self._map_point(p2.x, p2.y)
                    cv2.line(image, (x1, y1), (x2, y2), link_schema.color, 2)

    # 不再是 staticmethod，因为它需要调用 self._map_point
    def _draw_face_direction(self, image: np.ndarray, skeleton: ExpandedSkeleton):
        if skeleton.direction_type != FaceDirection.Unknown:
            # ★ 关键改动: 缩放原点
            origin_x, origin_y = self._map_point(skeleton.direction_origin[0], skeleton.direction_origin[1])

            # 方向向量不缩放
            vec_x, vec_y = skeleton.direction_vector
            end_x = int(skeleton.direction_modulus * vec_x + origin_x)
            end_y = int(skeleton.direction_modulus * vec_y + origin_y)
            cv2.arrowedLine(image, (origin_x, origin_y), (end_x, end_y),
                            (0, 255, 255), 2, tipLength=0.3)

    def _draw_label(self,
                    image: np.ndarray,
                    skeleton: ExpandedSkeleton,
                    label_map: Dict = None,
                    background_color: Tuple[int, int, int] = None,
                    append_track_id: bool = False):

        if label_map and skeleton.classification in label_map.keys():
            label = label_map[skeleton.classification]
        else:
            label = f"Class {skeleton.classification}"
        if append_track_id and skeleton.track_id is not None:
            label = f"#{skeleton.track_id} {label}"
        label = f"{label}-{skeleton.confidence:.2f}"

        # 使用新的映射函数
        x1, y1 = self._map_point(skeleton.rect.x1, skeleton.rect.y1)

        if background_color is None:
            bbox_colors = self.schema.bbox_colors
            bbox_color = bbox_colors[skeleton.classification % len(bbox_colors)]
        else:
            bbox_color = background_color

        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - h - 10), (x1 + w, y1), bbox_color, -1)
        text_painter.draw_text(image, label, (x1, y1 - h - 5), font_scale=0.5, thickness=1)

    def _draw_bbox_with_highlight(self, image: np.ndarray, skeleton: ExpandedSkeleton, highlight_classname: str):
        # 使用新的映射函数
        x1, y1, x2, y2 = self._map_rect(skeleton.rect)
        bbox_highlight_colors = self.schema.highlight_colors
        highlight_color = bbox_highlight_colors.get(highlight_classname, [[255, 0, 0], [255, 255, 255]])

        if self.blink_counter % 2 == 0:
            cv2.rectangle(image, (x1, y1), (x2, y2), highlight_color[1], 2)
        else:
            cv2.rectangle(image, (x1, y1), (x2, y2), highlight_color[0], 2)

    # ---------- 外部接口 ----------
    def draw_inference(self,
                       image: np.ndarray,
                       skeletons: List,
                       draw_bbox: bool = True,
                       draw_kpts: bool = True,
                       draw_links: bool = True,
                       draw_direction: bool = True,
                       label_map: Dict = None,
                       enable_track_id: bool = False,
                       bbox_color: Tuple[int, int, int] = None,
                       highlight_classes: List[Tuple[bool, str]] = None) -> np.ndarray:

        # 直接在传入的图像上绘制（不做自动缩放）
        display_image = image.copy()

        for idx, skeleton in enumerate(skeletons):
            if skeleton.confidence < self.bbox_conf_thresh:
                continue

            if draw_bbox:
                self._draw_bbox(display_image, skeleton, bbox_color=bbox_color)
            if draw_links:
                self._draw_skeleton_links(display_image, skeleton)
            if draw_kpts:
                self._draw_keypoints(display_image, skeleton)
            if draw_direction and isinstance(skeleton, ExpandedSkeleton):
                self._draw_face_direction(display_image, skeleton)
            if label_map:
                self._draw_label(display_image, skeleton, label_map,
                                 background_color=bbox_color,
                                 append_track_id=enable_track_id)
            if highlight_classes:
                enable_highlight, highlight_classname = highlight_classes[idx]
                if enable_highlight:
                    self._draw_bbox_with_highlight(display_image, skeleton, highlight_classname)

        self.blink_counter += 1
        return display_image

    def draw_bbox_with_custom_color(self,
                                    image: np.ndarray,
                                    skeleton: ExpandedSkeleton,
                                    bbox_color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:

        # 直接在传入的图像上绘制（不做自动缩放）
        display_image = image.copy()

        if skeleton.confidence < self.bbox_conf_thresh:
            return display_image

        self._draw_bbox(display_image, skeleton, bbox_color=bbox_color)
        return display_image