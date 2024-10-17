import numpy as np
from common.tracker.kalman_filter import KalmanFilter
from scipy.optimize import linear_sum_assignment



def iou(bb_test, bb_gt):
    xx1 = max(bb_test[0], bb_gt[0])
    yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[2], bb_gt[2])
    yy2 = min(bb_test[3], bb_gt[3])

    union_w = max(0., xx2 - xx1)
    union_h = max(0., yy2 - yy1)
    intersection = union_w * union_h
    area_bb_test = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
    area_bb_gt = (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])
    union = area_bb_test + area_bb_gt - intersection
    ratio = intersection / (union + 1e-6)
    return ratio


class Sort:
    def __init__(self, max_age=5, min_hits=1, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits  # 最小检测次数
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.next_id = 0

    def update(self, dets=np.empty((0, 4))):
        self.frame_count += 1

        # Get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            trks[t][:4] = trk.get_state()
            trks[t][4] = trk.id
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))

        # Remove dead tracklets
        for t in reversed(to_del):
            self.trackers.pop(t)

        matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(dets, trks)

        # Update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0]])

        # Create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanFilter(self.next_id, dets[i])
            self.next_id += 1
            self.trackers.append(trk)

        # Prepare return list
        ret = []
        for trk in self.trackers:
            if trk.time_since_update <= self.max_age:
                d = trk.get_state()
                ret.append((d, trk.id))

        # Remove dead tracklets
        self.trackers = [trk for trk in self.trackers if trk.time_since_update <= self.max_age]

        return ret

    def associate_detections_to_trackers(self, detections, trackers):
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

        iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                iou_matrix[d, t] = iou(det, trk[:4])

        matched_indices = linear_sum_assignment(-iou_matrix)
        matched_indices = np.array(list(zip(*matched_indices)))

        # 处理空的 matched_indices
        if matched_indices.size == 0:
            matched_indices = np.empty((0, 2), dtype=int)

        unmatched_detections = []
        for d in range(len(detections)):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)

        unmatched_trackers = []
        for t in range(len(trackers)):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)

        # Filter out matched with low IOU
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < self.iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))

        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)