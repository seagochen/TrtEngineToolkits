import numpy as np
from scipy.linalg import inv


class KalmanFilter:
    def __init__(self, id, bbox):
        # State vector [x, y, s, r, vx, vy, vs]
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2
        y = bbox[1] + h / 2
        s = w * h
        r = w / h
        self.x = np.array([x, y, s, r, 0, 0, 0], dtype=float)

        self.id = id
        self.time_since_update = 0
        self.hit_streak = 0

        # Define constant velocity model
        dt = 1  # Time step

        # State transition matrix for 7x7 state
        self.A = np.array([[1, 0, 0, 0, dt, 0, 0],
                           [0, 1, 0, 0, 0, dt, 0],
                           [0, 0, 1, 0, 0, 0, dt],
                           [0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 0, 1]])

        # Measurement matrix
        self.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 1, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0]])

        # Process noise covariance for 7x7 state
        self.Q = np.eye(7) * 0.01

        # Measurement noise covariance
        self.R = np.eye(4) * 0.1

        # Initial covariance matrix
        self.P = np.eye(7)

    def predict(self):
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q
        self.time_since_update += 1
        return self.x

    def update(self, bbox):
        # 计算测量值
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2
        y = bbox[1] + h / 2
        s = w * h
        r = w / h
        z = np.array([x, y, s, r], dtype=float)
        # 计算创新
        y_residual = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ inv(S)
        self.x = self.x + K @ y_residual
        self.P = (np.eye(len(self.x)) - K @ self.H) @ self.P
        self.time_since_update = 0
        self.hit_streak += 1

    def get_state(self):
        x = self.x[0]
        y = self.x[1]
        s = self.x[2]
        r = self.x[3]

        # 确保 s 和 r 为正值
        s = max(s, 1e-6)
        r = max(r, 1e-6)

        # 计算宽度和高度
        w = np.sqrt(s * r)
        h = s / w

        # 返回边界框坐标 [x1, y1, x2, y2]
        return [x - w / 2, y - h / 2, x + w / 2, y + h / 2]