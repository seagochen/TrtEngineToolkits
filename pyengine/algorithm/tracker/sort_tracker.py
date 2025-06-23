import numpy as np
from scipy.optimize import linear_sum_assignment

from pyengine.algorithm.tracker.matching import iou_batch  # Import matching utility
from pyengine.algorithm.tracker.track import Track  # Import unified Track class


class UnifiedSortTracker:
    """
    Simple Online and Realtime Tracking (SORT) algorithm implementation.

    Uses Kalman Filter for motion prediction and IOU for association.
    """
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        """
        Initializes the SORT tracker.

        Args:
            max_age (int): Maximum number of frames a track can be kept without updates.
            min_hits (int): Minimum number of consecutive hits to confirm a track.
            iou_threshold (float): Minimum IOU overlap required for association.
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []
        self._next_id = 0

    def _next_track_id(self) -> int:
        """Generates the next unique track ID."""
        self._next_id += 1
        return self._next_id

    def update(self, detections: np.ndarray) -> np.ndarray:
        """
        Updates the tracker state with new detections.

        Args:
            detections (np.ndarray): Array of detections for the current frame.
                Format: [x1, y1, x2, y2, conf?, class?, kpts...?].
                The first 4 elements must be the bounding box.

        Returns:
            np.ndarray: Array of active tracks.
                Format: [track_id, x1, y1, x2, y2, <original_detection_info>].
                Returns empty array if no tracks are active.
        """
        # 1. Predict state for existing tracks
        for track in self.tracks:
            track.predict()

        # 2. Separate tracks into confirmed and tentative
        confirmed_tracks = [t for t in self.tracks if t.is_confirmed()]
        unconfirmed_tracks = [t for t in self.tracks if t.is_tentative()] # Not used in basic SORT matching

        # 3. Associate detections with confirmed tracks using IOU
        matched_indices = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(confirmed_tracks)))

        if len(detections) > 0 and len(confirmed_tracks) > 0:
            # Calculate IOU cost matrix
            track_bboxes = np.array([t.get_state() for t in confirmed_tracks])
            detection_bboxes = detections[:, :4]
            iou_matrix = iou_batch(detection_bboxes, track_bboxes)

            # Use Hungarian algorithm for assignment (maximize IOU = minimize -IOU)
            row_ind, col_ind = linear_sum_assignment(-iou_matrix)

            # Filter matches based on IOU threshold
            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] >= self.iou_threshold:
                    matched_indices.append((r, c)) # (det_idx, confirmed_track_idx)
                    # Remove from unmatched lists
                    if r in unmatched_detections: unmatched_detections.remove(r)
                    if c in unmatched_tracks: unmatched_tracks.remove(c)

        # 4. Update matched tracks
        # Store mapping from track_id to original detection info for output formatting
        output_info_map = {}
        for det_idx, track_idx in matched_indices:
            track = confirmed_tracks[track_idx]
            track.update(detections[det_idx, :4])
            # Store original detection info (e.g., score, class, keypoints) if available
            output_info_map[track.track_id] = detections[det_idx, 4:]

        # 5. Handle unmatched tracks (mark as missed) and tentative tracks
        for track_idx in unmatched_tracks:
            track = confirmed_tracks[track_idx]
            track.mark_missed()
        # Also mark tentative tracks that weren't used (basic SORT doesn't rematch tentative)
        for track in unconfirmed_tracks:
             track.mark_missed() # This will delete them if missed

        # 6. Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            bbox = detections[det_idx, :4]
            new_track = Track(bbox, self._next_track_id(), self.min_hits, self.max_age)
            self.tracks.append(new_track)
            # Store original detection info for potential immediate output if needed
            output_info_map[new_track.track_id] = detections[det_idx, 4:] # Store even for new track


        # 7. Remove deleted tracks and prepare output
        active_tracks_output = []
        next_tracks = []
        for track in self.tracks:
            if not track.is_deleted():
                next_tracks.append(track) # Keep track for next frame
                # Only output confirmed tracks (or tentative ones based on min_hits logic if desired)
                if track.is_confirmed(): # Basic SORT often outputs confirmed tracks
                # Or maybe output if hits >= min_hits?
                # if track.hits >= self.min_hits and track.time_since_update == 0:
                    state = track.get_state()
                    if not np.isnan(state).any(): # Check if KF state is valid
                        track_id = track.track_id
                        # Retrieve original detection info if available for this frame
                        original_info = output_info_map.get(track_id, [])
                        active_tracks_output.append(
                            np.concatenate(([track_id], state, original_info)).flatten()
                        )

        self.tracks = next_tracks

        if len(active_tracks_output) > 0:
            return np.stack(active_tracks_output)
        else:
            return np.empty((0, 5)) # Return empty array with at least track_id + bbox shape