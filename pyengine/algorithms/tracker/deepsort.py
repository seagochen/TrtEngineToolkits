# deepsort.py
from typing import List, Dict

import numpy as np
from scipy.optimize import linear_sum_assignment

# 从你的文件导入数据结构
from pyengine.inference.unified_structs.inference_results import ObjectDetection, Rect
from pyengine.algorithms.tracker.tracker import UnifiedTrack


class DeepSORTTracker:
    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.5, reid_threshold: float = 0.4):
        self.tracks: List[UnifiedTrack] = []
        self.max_age = max_age  # Max frames for a track to be unmatched before deletion
        self.min_hits = min_hits  # Min hits for a new track to be confirmed
        self.iou_threshold = iou_threshold
        self.reid_threshold = reid_threshold  # Cosine similarity threshold for re-ID

    def _iou_cost(self, track_bbox: Rect, det_bbox: Rect) -> float:
        # Calculate IoU distance (1 - IoU)

        # Convert Rect to (x1, y1, x2, y2)
        bb1 = [track_bbox.x1, track_bbox.y1, track_bbox.x2, track_bbox.y2]
        bb2 = [det_bbox.x1, det_bbox.y1, det_bbox.x2, det_bbox.y2]

        x_left = max(bb1[0], bb2[0])
        y_top = max(bb1[1], bb2[1])
        x_right = min(bb1[2], bb2[2])
        y_bottom = min(bb1[3], bb2[3])

        if x_right < x_left or y_bottom < y_top:
            return 1.0  # No overlap, distance is 1 (max)

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
        bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
        return 1.0 - iou  # Return 1 - IoU as distance/cost

    def _cosine_distance(self, track_feature: np.ndarray, det_feature: np.ndarray) -> float:
        # Calculate cosine distance (1 - cosine_similarity)
        # Ensure features are normalized if not already
        track_feature = track_feature / np.linalg.norm(track_feature)
        det_feature = det_feature / np.linalg.norm(det_feature)

        similarity = np.dot(track_feature, det_feature)
        return 1.0 - similarity  # Return 1 - similarity as distance/cost

    def update(self, detections: List[ObjectDetection]) -> Dict[int, Rect]:
        # 1. Predict
        for track in self.tracks:
            track.predict()

        # 2. Separate confirmed and unconfirmed tracks (DeepSORT's cascading logic)
        # For simplicity here, we'll treat all tracks equally,
        # A full DeepSORT would prioritize matching to recent tracks first.

        # 3. Create cost matrix
        num_tracks = len(self.tracks)
        num_dets = len(detections)

        cost_matrix = np.full((num_tracks, num_dets), np.inf)

        for i, track in enumerate(self.tracks):
            track_predicted_bbox = track.get_state()
            track_feature = track.get_feature()

            for j, det in enumerate(detections):
                # IoU cost
                # det.box 不存在，现在 det 是 ObjectDetection，其边界框信息是 det.rect
                det_bbox_rect = det.rect # 直接使用 ObjectDetection 内部的 Rect 对象
                iou_cost = self._iou_cost(track_predicted_bbox, det_bbox_rect)

                # Re-ID cost (only if feature exists)
                reid_cost = np.inf
                # det.features 现在是 List[float]，需要将其转换为 np.ndarray
                if track_feature is not None and det.features:
                    reid_cost = self._cosine_distance(track_feature, np.array(det.features))

                # Combine costs (DeepSORT's typical combination)
                # You'd tune weights or use gating for these
                if iou_cost <= (1.0 - self.iou_threshold) and reid_cost <= (1.0 - self.reid_threshold):
                    # Example weighted sum. DeepSORT uses a more sophisticated approach
                    # with gating on Mahalanobis distance as well.
                    cost_matrix[i, j] = 0.5 * iou_cost + 0.5 * reid_cost
                else:
                    cost_matrix[i, j] = np.inf  # Not compatible

        # 4. Solve Assignment (Hungarian Algorithm)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matches = []
        unmatched_tracks_indices = set(range(num_tracks))
        unmatched_dets_indices = set(range(num_dets))

        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < np.inf:  # Check if it's a valid match
                matches.append((r, c))
                unmatched_tracks_indices.discard(r)
                unmatched_dets_indices.discard(c)

        # 5. Update Tracks
        for track_idx, det_idx in matches:
            self.tracks[track_idx].update(detections[det_idx])

        # Handle unmatched tracks
        for track_idx in unmatched_tracks_indices:
            self.tracks[track_idx].time_since_update += 1

        # Remove old, unmatched tracks
        self.tracks = [track for track in self.tracks if track.time_since_update <= self.max_age]

        # 6. Initialize New Tracks
        for det_idx in unmatched_dets_indices:
            # 只有当检测的置信度高于某个阈值时，才考虑初始化新轨迹
            # 使用 ObjectDetection 的 confidence 字段
            if detections[det_idx].confidence >= 0.5:  # 假设一个合理的置信度
                new_track = UnifiedTrack(detections[det_idx])  # DeepSORTTracker 默认 use_reid=True
                self.tracks.append(new_track)

        # Filter out "tentative" new tracks that haven't been seen enough times
        # This is part of DeepSORT's `min_hits` logic
        final_tracks_output = {}
        for track in self.tracks:
            if track.hits >= self.min_hits or track.age < self.min_hits:  # Simplified confirmation
                final_tracks_output[track.track_id] = track.get_state()

        return final_tracks_output

