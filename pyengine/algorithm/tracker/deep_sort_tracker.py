# unified_deepsort.py
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch # Needed if feature_extractor is PyTorch model

from track import Track, TrackState # Import unified Track class
from matching import iou_batch, cosine_distance # Import matching utilities


class UnifiedDeepSortTracker:
    """
    DeepSORT algorithm implementation using appearance features and motion.

    Args:
        feature_extractor (torch.nn.Module): Model to extract appearance features.
        max_dist (float): Maximum cosine distance for appearance matching.
        iou_threshold (float): Minimum IOU for secondary matching cascade.
        max_age (int): Maximum number of frames without association.
        n_init (int): Minimum number of hits to confirm a track.
        nn_budget (Optional[int]): Maximum size of the appearance feature gallery for
                                    each track. If None, all features are stored.
                                    Corresponds to max_features_history in Track.
        device (str or torch.device): Device for feature extraction ('cuda' or 'cpu').
    """
    def __init__(self, feature_extractor: torch.nn.Module,
                 max_dist: float = 0.2,
                 iou_threshold: float = 0.7,
                 max_age: int = 70,
                 n_init: int = 3,
                 nn_budget: int = 100,
                 device = 'cuda' if torch.cuda.is_available() else 'cpu'):

        self.feature_extractor = feature_extractor
        if isinstance(self.feature_extractor, torch.nn.Module):
             self.feature_extractor.to(device)
             self.feature_extractor.eval() # Set to evaluation mode
        self.device = device

        self.max_dist = max_dist
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.n_init = n_init
        self.nn_budget = nn_budget # Controls feature history size in Track

        self.tracks = []
        self._next_id = 0

    def _next_track_id(self) -> int:
        """Generates the next unique track ID."""
        self._next_id += 1
        return self._next_id

    def _get_features(self, image_patches: list) -> np.ndarray:
        """Extracts features from image patches."""
        if not image_patches:
            return np.empty((0, self.feature_extractor.fc.out_features)) # Adjust based on your model's output dim

        # Assuming patches are numpy arrays [H, W, C] or PIL Images
        # Preprocess patches according to your feature extractor's requirements
        # Example preprocessing (adjust for your model):
        from torchvision import transforms
        preprocess = transforms.Compose([
            transforms.ToPILImage(), # If input is numpy array
            transforms.Resize((224, 224)), # Example resize
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        try:
            patch_tensors = torch.stack([preprocess(patch) for patch in image_patches]).to(self.device)
        except Exception as e:
             print(f"Error preprocessing patches: {e}")
             # Return empty or handle error appropriately
             return np.empty((0, self.feature_extractor.fc.out_features))


        with torch.no_grad():
            features = self.feature_extractor(patch_tensors) # Run model

        return features.cpu().numpy()

    def _match(self, detections, det_features):
        """Performs the matching cascade."""

        # Split tracks by state
        confirmed_tracks = [t for t in self.tracks if t.is_confirmed()]
        unconfirmed_tracks = [t for t in self.tracks if t.is_tentative()]

        matches, unmatched_detections_indices, unmatched_tracks_indices = \
            self._match_appearance(confirmed_tracks, detections, det_features)

        # Match remaining tracks and detections using IOU
        # Use tentative tracks + unmatched confirmed tracks
        iou_track_candidates_indices = [i for i, t in enumerate(self.tracks)
                                         if t.is_tentative() or (t.is_confirmed() and i in unmatched_tracks_indices)]
        iou_track_candidates = [self.tracks[i] for i in iou_track_candidates_indices]

        unmatched_detections_indices_iou = unmatched_detections_indices # Indices into original detections
        # Note: unmatched_tracks_indices are indices into the original self.tracks list

        if len(iou_track_candidates) > 0 and len(unmatched_detections_indices_iou) > 0:
            # Get bboxes for IOU matching
            track_bboxes = np.array([iou_track_candidates[i].get_state() for i in range(len(iou_track_candidates))])
            detection_bboxes = np.array([detections[i][:4] for i in unmatched_detections_indices_iou])

            iou_matrix = iou_batch(detection_bboxes, track_bboxes)

            row_ind, col_ind = linear_sum_assignment(-iou_matrix)

            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] >= self.iou_threshold:
                    det_idx = unmatched_detections_indices_iou[r]
                    # Map col_ind 'c' (index within iou_track_candidates) back to original self.tracks index
                    track_idx_original = iou_track_candidates_indices[c]

                    matches.append((det_idx, track_idx_original))
                    # Remove from respective unmatched lists if they exist there
                    if det_idx in unmatched_detections_indices: unmatched_detections_indices.remove(det_idx)
                    if track_idx_original in unmatched_tracks_indices: unmatched_tracks_indices.remove(track_idx_original)


        # Final list of unmatched detection indices (relative to original detections)
        final_unmatched_det_indices = unmatched_detections_indices
        # Final list of unmatched track indices (relative to original self.tracks)
        final_unmatched_track_indices = unmatched_tracks_indices

        return matches, final_unmatched_det_indices, final_unmatched_track_indices


    def _match_appearance(self, tracks, detections, features):
        """Matches tracks and detections based on appearance features."""
        matches = []
        unmatched_track_indices = list(range(len(tracks)))
        unmatched_detection_indices = list(range(len(detections)))

        if len(detections) == 0 or len(tracks) == 0:
            return matches, unmatched_detection_indices, unmatched_track_indices

        # Calculate cosine distance cost matrix
        track_features = np.stack([t.get_last_feature() for t in tracks if t.get_last_feature() is not None], axis=0)

        # Handle case where some tracks might not have features yet?
        valid_track_indices = [i for i, t in enumerate(tracks) if t.get_last_feature() is not None]
        if len(valid_track_indices) == 0: # No tracks with features
             return matches, unmatched_detection_indices, unmatched_track_indices

        track_features = np.stack([tracks[i].get_last_feature() for i in valid_track_indices], axis=0)


        cost_matrix = cosine_distance(features, track_features)

        # Gating with Mahalanobis distance could be added here for stricter matching

        # Hungarian assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        for r, c in zip(row_ind, col_ind):
            # Map 'c' back to the original index in 'tracks' list
            track_idx_original = valid_track_indices[c]
            if cost_matrix[r, c] < self.max_dist:
                matches.append((r, track_idx_original)) # (det_idx, original_track_idx)
                # Remove from unmatched lists
                if r in unmatched_detection_indices: unmatched_detection_indices.remove(r)
                if track_idx_original in unmatched_track_indices: unmatched_track_indices.remove(track_idx_original)

        return matches, unmatched_detection_indices, unmatched_track_indices


    def update(self, detections: np.ndarray, image: np.ndarray):
        """
        Updates the tracker state with new detections and image frame.

        Args:
            detections (np.ndarray): Detections [N, 4+], (x1,y1,x2,y2,...).
            image (np.ndarray): The full image frame [H, W, C].

        Returns:
            np.ndarray: Active tracks [track_id, x1, y1, x2, y2, <original_detection_info>].
        """
        # 1. Extract features for current detections
        detection_bboxes = detections[:, :4].astype(int)
        # Clip bboxes to image bounds before cropping
        h, w = image.shape[:2]
        detection_bboxes[:, [0, 2]] = detection_bboxes[:, [0, 2]].clip(0, w - 1)
        detection_bboxes[:, [1, 3]] = detection_bboxes[:, [1, 3]].clip(0, h - 1)
        # Filter out invalid bboxes after clipping (width/height <= 0)
        valid_indices = (detection_bboxes[:, 2] > detection_bboxes[:, 0]) & (detection_bboxes[:, 3] > detection_bboxes[:, 1])
        valid_detections = detections[valid_indices]
        valid_bboxes = detection_bboxes[valid_indices]

        if len(valid_detections) == 0:
             det_features = np.empty((0,0)) # Handle no valid detections case
        else:
            image_patches = [image[b[1]:b[3], b[0]:b[2]] for b in valid_bboxes]
            # Filter out empty patches if bbox was invalid somehow after clipping
            valid_patches_indices = [i for i,p in enumerate(image_patches) if p.size > 0]
            if len(valid_patches_indices) < len(image_patches):
                print(f"Warning: Some bounding boxes resulted in empty patches after cropping.")
                image_patches = [image_patches[i] for i in valid_patches_indices]
                valid_detections = valid_detections[valid_patches_indices]
                # Update valid_indices to reflect only those with valid patches
                original_indices = np.where(valid_indices)[0]
                valid_indices = original_indices[valid_patches_indices]


            if not image_patches: # Check if list is empty after filtering
                det_features = np.empty((0,0))
            else:
                det_features = self._get_features(image_patches)

        # Map det_features back to original detection indices if filtering occurred
        mapped_det_features = np.zeros((len(detections), det_features.shape[1])) if det_features.size > 0 else np.zeros((len(detections), 0))
        if det_features.size > 0:
             mapped_det_features[valid_indices] = det_features
        det_features = mapped_det_features # Use mapped features aligned with original detections


        # 2. Predict track states
        for track in self.tracks:
            track.predict()

        # 3. Perform matching
        # Use only valid detections for matching
        matches, unmatched_det_indices, unmatched_track_indices = \
            self._match(valid_detections, det_features[valid_indices] if det_features.size > 0 else np.empty((0,0)))

        # Map matched indices back to original detection indices
        original_matches = []
        output_info_map = {}
        original_valid_indices = np.where(valid_indices)[0] # Get original indices of valid detections
        for det_idx_valid, track_idx in matches:
             original_det_idx = original_valid_indices[det_idx_valid]
             original_matches.append((original_det_idx, track_idx))
             # Store mapping from track_id to original detection info
             output_info_map[self.tracks[track_idx].track_id] = detections[original_det_idx, 4:]

        # Map unmatched detection indices back to original indices
        final_unmatched_original_det_indices = [original_valid_indices[i] for i in unmatched_det_indices]


        # 4. Update matched tracks
        for original_det_idx, track_idx in original_matches:
            feature = det_features[original_det_idx] if det_features.size > 0 else None
            self.tracks[track_idx].update(detections[original_det_idx, :4], feature)

        # 5. Handle unmatched tracks
        for track_idx in unmatched_track_indices:
             self.tracks[track_idx].mark_missed()

        # 6. Create new tracks for unmatched detections
        for original_det_idx in final_unmatched_original_det_indices:
            bbox = detections[original_det_idx, :4]
            feature = det_features[original_det_idx] if det_features.size > 0 else None
            new_track = Track(bbox, self._next_track_id(), self.n_init, self.max_age,
                              feature, max_features_history=self.nn_budget)
            self.tracks.append(new_track)
            # Store original detection info for potential immediate output
            output_info_map[new_track.track_id] = detections[original_det_idx, 4:]

        # 7. Remove deleted tracks and prepare output
        active_tracks_output = []
        next_tracks = []
        for track in self.tracks:
            if not track.is_deleted():
                next_tracks.append(track)
                # Output confirmed tracks only
                if track.is_confirmed():
                    state = track.get_state()
                    if not np.isnan(state).any():
                         track_id = track.track_id
                         original_info = output_info_map.get(track_id, [])
                         active_tracks_output.append(
                            np.concatenate(([track_id], state, original_info)).flatten()
                         )

        self.tracks = next_tracks

        if len(active_tracks_output) > 0:
            return np.stack(active_tracks_output)
        else:
            return np.empty((0, 5)) # track_id + bbox