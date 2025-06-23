# track.py
import numpy as np
from pyengine.algorithm.tracker.kalman_filter import KalmanFilter # Import the unified KF

class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    `Tentative` until enough evidence has been collected. Then they become
    `Confirmed`. Tracks that are no longer alive are `Deleted`.
    """
    Tentative = 1
    Confirmed = 2
    Deleted = 3

class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Args:
        initial_bbox (list or np.array): Bounding box [x1, y1, x2, y2].
        track_id (int): A unique track ID.
        n_init (int): Number of consecutive detections before the track is confirmed.
                      The track state is set to `Deleted` if a miss occurs within
                      the first `n_init` frames.
        max_age (int): The maximum number of consecutive misses before the track state
                       is set to `Deleted`.
        feature (Optional[np.ndarray]): The first appearance feature of the object.
        max_features_history (int): Max number of appearance features to store.
    """

    def __init__(self, initial_bbox, track_id: int, n_init: int, max_age: int,
                 feature: np.ndarray = None, max_features_history: int = 100):

        self.track_id = track_id
        self.hits = 1           # Number of consecutive updates (matches)
        self.age = 1            # Total number of frames since first occurrence
        self.time_since_update = 0 # Number of frames since last update

        self.state = TrackState.Tentative # Initial state
        self._n_init = n_init
        self._max_age = max_age
        self._max_features_history = max_features_history

        self.kf = KalmanFilter(initial_bbox)
        self.features = []      # List to store appearance features
        if feature is not None:
            # Normalize the feature? DeepSORT often uses L2 normalization.
            # feature /= np.linalg.norm(feature)
            self.features.append(feature)
            if len(self.features) > self._max_features_history:
                 self.features.pop(0) # Keep fixed size history


    def predict(self):
        """Propagate the state distribution to the current time step using KF."""
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, detection_bbox, feature: np.ndarray = None):
        """
        Perform measurement update and track state management.

        Args:
            detection_bbox (list or np.array): The measurement bbox [x1, y1, x2, y2].
            feature (Optional[np.ndarray]): The associated appearance feature.
        """
        self.kf.update(detection_bbox)

        self.hits += 1
        self.time_since_update = 0

        # Update state based on hits
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

        # Update features (only if provided - relevant for DeepSORT)
        if feature is not None:
            # Normalize?
            # feature /= np.linalg.norm(feature)
            self.features.append(feature)
            if len(self.features) > self._max_features_history:
                 self.features.pop(0) # Remove oldest feature


    def mark_missed(self):
        """Mark that the track was not associated with any detection."""
        # For tentative tracks, delete immediately if missed before confirmation
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        # For confirmed tracks, check if max_age is exceeded
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (recently created)."""
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted

    def get_state(self):
        """
        Returns the current bounding box estimate `[x1, y1, x2, y2]`.
        """
        return self.kf.get_state()

    def get_feature_history(self):
        """Returns the list of stored appearance features."""
        return self.features

    def get_last_feature(self):
        """Returns the most recent appearance feature."""
        return self.features[-1] if self.features else None