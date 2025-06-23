# kalman_filter.py
import numpy as np
# from scipy.linalg import inv # Using pinv now for stability
import scipy.linalg # Keep for potential future use or comparison

class KalmanFilter:
    """
    A simple Kalman filter implementation for tracking bounding boxes.
    Assumes a state space [cx, cy, s, r, vx, vy, vs] where:
        cx, cy: center coordinates
        s: area (width * height)
        r: aspect ratio (width / height)
        vx, vy, vs: velocities for cx, cy, s
    Assumes constant velocity model. Aspect ratio velocity is optional (currently off).
    """
    def __init__(self, initial_bbox):
        """
        Initializes the Kalman filter.

        Args:
            initial_bbox (list or np.array): Initial bounding box [x1, y1, x2, y2].
        """
        # State: [cx, cy, s, r, vx, vy, vs]
        x1, y1, x2, y2 = initial_bbox
        w, h = x2 - x1, y2 - y1
        # Safeguard for zero width/height during initialization
        if w <= 0: w = 1e-6
        if h <= 0: h = 1e-6
        cx, cy = x1 + w / 2, y1 + h / 2
        s = w * h
        r = w / (h + 1e-6) # Avoid division by zero if h is extremely small
        # Initialize state with zero velocities
        self.x = np.array([cx, cy, s, r, 0, 0, 0], dtype=float)

        # State transition matrix (A) - Constant Velocity Model
        self.A = np.eye(7)
        dt = 1.0 # Time step assumed to be 1 frame
        self.A[0, 4] = dt # cx += vx * dt
        self.A[1, 5] = dt # cy += vy * dt
        self.A[2, 6] = dt # s += vs * dt
        # Aspect ratio (r) velocity is often ignored or modeled differently, keeping it constant here.

        # Measurement matrix (H) - We observe cx, cy, s, r directly from bbox
        self.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ])

        # Process noise covariance (Q) - Uncertainty in the motion model (tuneable)
        # Higher values = more uncertainty about motion prediction
        std_pos = 1. / 20
        std_vel = 1. / 160
        self.Q = np.diag([
            std_pos * (w**2), # cx noise variance proportional to w^2 ?
            std_pos * (h**2), # cy noise variance proportional to h^2 ?
            1e-2,           # Area (s) noise variance
            1e-5,           # Aspect ratio (r) noise variance (usually stable)
            std_vel * (w**2), # vx noise variance ?
            std_vel * (h**2), # vy noise variance ?
            1e-4            # vs noise variance
        ])
        # Or simpler fixed values:
        # self.Q = np.eye(7)
        # self.Q[0,0] = self.Q[1,1] = 0.01 # cx, cy
        # self.Q[2,2] = 1e-4             # s
        # self.Q[3,3] = 1e-6             # r
        # self.Q[4,4] = self.Q[5,5] = 0.001 # vx, vy
        # self.Q[6,6] = 1e-5             # vs

        # Measurement noise covariance (R) - Uncertainty in the detection (tuneable)
        # Higher values = more uncertainty about detection accuracy
        std_meas_pos_factor = 1. / 150 # Factor relative to size
        std_meas_size_factor = 1. / 100
        self.R = np.diag([
            std_meas_pos_factor * w, # cx measurement noise (proportional to w?)
            std_meas_pos_factor * h, # cy measurement noise (proportional to h?)
            std_meas_size_factor * s,  # Area measurement noise (proportional to s?)
            0.1                      # Aspect ratio measurement noise (fixed?)
        ])**2 # Variance is std^2
        # Or simpler fixed values:
        # self.R = np.diag([0.1**2, 0.1**2, 1**2, 0.1**2]) # Example: Var(cx,cy)=0.01, Var(s)=1, Var(r)=0.01


        # State covariance matrix (P) - Estimate uncertainty
        self.P = np.eye(7) * 1.0 # Start with moderate uncertainty
        # Increase initial velocity uncertainty if needed
        # self.P[4:,4:] *= 100.0

    def predict(self):
        """Predict the next state and update uncertainty."""
        self.x = self.A @ self.x
        # Ensure state remains physically plausible (optional but recommended)
        self.x[2] = max(1e-6, self.x[2]) # Area (s) > 0
        self.x[3] = max(1e-6, self.x[3]) # Aspect ratio (r) > 0

        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x

    def update(self, measurement_bbox):
        """Update the state estimate with a new measurement."""
        x1, y1, x2, y2 = measurement_bbox
        w, h = x2 - x1, y2 - y1
        # Safeguard measurement
        if w <= 0: w = 1e-6
        if h <= 0: h = 1e-6
        cx = x1 + w / 2
        y = y1 + h / 2
        s = w * h
        r = w / (h + 1e-6)
        z = np.array([cx, y, s, r], dtype=float) # Measurement vector

        # Measurement residual (innovation)
        y_residual = z - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain (using pseudo-inverse for stability)
        try:
            K = self.P @ self.H.T @ np.linalg.pinv(S)
        except np.linalg.LinAlgError:
            # Handle case where inversion fails
            print("Warning: Kalman gain calculation failed (LinAlgError). Skipping update.")
            # Optionally increase uncertainty P significantly here
            # self.P += np.eye(7) * 0.1
            return # Skip the rest of the update

        # State update
        self.x = self.x + K @ y_residual

        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(7) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

        # Ensure state remains physically plausible after update
        self.x[2] = max(1e-6, self.x[2]) # Ensure area 's' is positive
        self.x[3] = max(1e-6, self.x[3]) # Ensure aspect ratio 'r' is positive

        # Check for NaNs after update (optional debug)
        # if np.isnan(self.x).any() or np.isnan(self.P).any():
        #     print(f"Warning: NaN detected in KF state/covariance after update!")
            # Consider resetting P or taking other action?


    def get_state(self):
        """Returns the current estimated bounding box [x1, y1, x2, y2]."""
        # Check for NaN in internal state before calculation
        if np.isnan(self.x[:4]).any():
             # print("Warning: NaN detected in Kalman state x[:4] before get_state calculation.")
             return [np.nan] * 4 # Return NaNs if state is already bad

        cx, cy, s, r = self.x[:4]

        # Safeguard state values before calculations
        s = max(1e-6, s) # Ensure area is positive
        r = max(1e-6, r) # Ensure ratio is positive

        # Recalculate w, h from s, r
        w = np.sqrt(s * r)
        # h = s / (w + 1e-6) # Add epsilon for safety
        # Avoid potential division by zero if w becomes zero despite safeguards
        if w < 1e-6:
            h = np.sqrt(s / (r + 1e-6)) if r > 1e-6 else np.sqrt(s) # Estimate h from s if r is also tiny
        else:
            h = s / w

        # Ensure w and h are not NaN/inf before calculating bbox
        if not np.isfinite(w) or not np.isfinite(h) or w <= 0 or h <= 0:
            # print(f"Warning: Non-finite/non-positive w ({w}) or h ({h}) in get_state. State x[:4]={self.x[:4]}")
            # Attempt to return centroid if size calculation fails?
            # Or return NaN as before
            return [np.nan] * 4 # Return NaNs

        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return [x1, y1, x2, y2]