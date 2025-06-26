# sort.py
from typing import List, Dict

import numpy as np
from scipy.optimize import linear_sum_assignment

# 从你的数据结构模块导入 ObjectDetection 和 Rect
from pyengine.inference.unified_structs.inference_results import ObjectDetection, Rect
from pyengine.algorithms.tracker.tracker import UnifiedTrack


class SORTTracker:
    def __init__(self, max_age: int = 1, min_hits: int = 3, iou_threshold: float = 0.3):
        """
        初始化 SORT 追踪器。
        Args:
            max_age (int): 轨迹在未更新后被删除的最大帧数。
                           SORT通常设置为1或2，因为不使用Re-ID，长时间丢失意味着真正丢失。
            min_hits (int): 新轨迹在被确认为有效轨迹前的最小命中次数。
            iou_threshold (float): 用于匹配的 IoU 阈值。
        """
        self.tracks: List[UnifiedTrack] = []
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.use_reid = False

    def _calculate_iou(self, bbox1: Rect, bbox2: Rect) -> float:
        """
        计算两个边界框的 IoU。
        Args:
            bbox1 (Rect): 第一个边界框。
            bbox2 (Rect): 第二个边界框。
        Returns:
            float: IoU 值。
        """
        # Rect 已经是 (x1, y1, x2, y2) 形式，直接访问属性
        bb1 = [bbox1.x1, bbox1.y1, bbox1.x2, bbox1.y2]
        bb2 = [bbox2.x1, bbox2.y1, bbox2.x2, bbox2.y2]

        x_left = max(bb1[0], bb2[0])
        y_top = max(bb1[1], bb2[1])
        x_right = min(bb1[2], bb2[2])
        y_bottom = min(bb1[3], bb2[3])

        if x_right < x_left or y_bottom < y_top:
            return 0.0  # No overlap

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
        bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

        union_area = bb1_area + bb2_area - intersection_area
        if union_area == 0:
            return 0.0  # Avoid division by zero

        iou = intersection_area / union_area
        return iou

    def update(self, detections: List[ObjectDetection]) -> Dict[int, Rect]:
        """
        更新追踪器状态，处理当前帧的检测结果。
        Args:
            detections (List[ObjectDetection]): 当前帧的检测结果列表。
        Returns:
            Dict[int, Rect]: 字典，键是轨迹ID，值是对应轨迹的边界框。
                             只包含被确认的或未达到min_hits但还在追踪中的轨迹。
        """
        # 1. 预测现有轨迹状态
        for track in self.tracks:
            track.predict()

        # 2. 构建 IoU 成本矩阵
        num_tracks = len(self.tracks)
        num_dets = len(detections)

        # 成本矩阵 (cost = 1 - IoU)，如果 IoU 低于阈值，则成本为无穷大
        cost_matrix = np.full((num_tracks, num_dets), np.inf)

        for i, track in enumerate(self.tracks):
            # 获取 Kalman 滤波器预测的边界框
            track_predicted_bbox = track.get_state()

            for j, det in enumerate(detections):
                # det.box 不存在，现在 det 是 ObjectDetection，其边界框信息是 det.rect
                det_bbox_rect = det.rect # 直接使用 ObjectDetection 内部的 Rect 对象
                iou = self._calculate_iou(track_predicted_bbox, det_bbox_rect)
                if iou >= self.iou_threshold:
                    cost_matrix[i, j] = 1.0 - iou  # IoU 越高，成本越低

        # 3. 使用匈牙利算法进行匹配
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matches = []
        unmatched_tracks_indices = set(range(num_tracks))
        unmatched_dets_indices = set(range(num_dets))

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < np.inf:  # 确保匹配是有效的（IoU在阈值之上）
                matches.append((r, c))
                unmatched_tracks_indices.discard(r)
                unmatched_dets_indices.discard(c)

        # 4. 更新轨迹状态
        for track_idx, det_idx in matches:
            self.tracks[track_idx].update(detections[det_idx])

        # 5. 处理未匹配的轨迹
        # 对于未匹配的现有轨迹，增加 time_since_update 计数
        for track_idx in unmatched_tracks_indices:
            self.tracks[track_idx].time_since_update += 1

        # 6. 删除老旧的轨迹 (超过 max_age 未更新)
        self.tracks = [track for track in self.tracks if not track.is_deleted(self.max_age)]

        # 7. 初始化新轨迹
        for det_idx in unmatched_dets_indices:
            # 只有当检测的置信度高于某个阈值时，才考虑初始化新轨迹
            # 使用 ObjectDetection 的 confidence 字段
            if detections[det_idx].confidence >= 0.5:  # 假设一个合理的置信度，可以作为 SORT 参数
                new_track = UnifiedTrack(detections[det_idx], use_reid=self.use_reid)
                self.tracks.append(new_track)

        # 8. 收集并返回当前帧的有效跟踪结果
        # SORT 只有在轨迹被“命中”足够次数 (min_hits) 后才输出它
        # 这可以防止因单次误检测而创建大量假轨迹
        final_tracked_objects = {}
        for track in self.tracks:
            # 只有被确认的轨迹才会被输出。
            # 或者，如果 max_age 设置为 1，那么只要被命中过就会被输出。
            if track.is_confirmed(self.min_hits) or (track.time_since_update == 0 and track.hits == 1):
                final_tracked_objects[track.track_id] = track.get_state()

        return final_tracked_objects