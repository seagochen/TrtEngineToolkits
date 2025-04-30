# matching.py
import numpy as np

def iou(bbox1, bbox2):
    """
    Calculates the Intersection over Union (IoU) between two bounding boxes.

    Args:
        bbox1 (list or np.array): [x1, y1, x2, y2]
        bbox2 (list or np.array): [x1, y1, x2, y2]

    Returns:
        float: The IoU value (between 0 and 1).
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # Calculate intersection coordinates
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)

    # Calculate intersection area
    inter_width = max(0., x2_inter - x1_inter)
    inter_height = max(0., y2_inter - y1_inter)
    intersection_area = inter_width * inter_height

    # Calculate union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - intersection_area

    # Calculate IoU
    iou_value = intersection_area / (union_area + 1e-6) # Add epsilon for stability
    return max(0., iou_value) # Ensure IoU is not negative due to float errors

def iou_batch(bboxes1, bboxes2):
    """
    Calculates the IoU matrix between two sets of bounding boxes.

    Args:
        bboxes1 (np.array): Shape (N, 4), format [x1, y1, x2, y2].
        bboxes2 (np.array): Shape (M, 4), format [x1, y1, x2, y2].

    Returns:
        np.array: IoU matrix of shape (N, M).
    """
    N = bboxes1.shape[0]
    M = bboxes2.shape[0]
    iou_matrix = np.zeros((N, M), dtype=np.float32)

    for i in range(N):
        for j in range(M):
            iou_matrix[i, j] = iou(bboxes1[i], bboxes2[j])

    return iou_matrix

def cosine_distance(features1, features2, epsilon=1e-12):
    """
    Calculates the cosine distance matrix between two sets of feature vectors.
    Cosine Distance = 1 - Cosine Similarity

    Args:
        features1 (np.array): Shape (N, F), feature vectors. Assumes L2 normalized.
        features2 (np.array): Shape (M, F), feature vectors. Assumes L2 normalized.
        epsilon (float): Small value to avoid division by zero.

    Returns:
        np.array: Cosine distance matrix of shape (N, M). Values range [0, 2].
                  Lower values indicate higher similarity.
    """
    # Ensure features are L2 normalized (important for cosine distance)
    features1_norm = features1 / (np.linalg.norm(features1, axis=1, keepdims=True) + epsilon)
    features2_norm = features2 / (np.linalg.norm(features2, axis=1, keepdims=True) + epsilon)


    # Calculate cosine similarity (dot product of normalized vectors)
    similarity_matrix = np.dot(features1_norm, features2_norm.T)

    # Clamp similarity to [-1, 1] to avoid potential float errors
    similarity_matrix = np.clip(similarity_matrix, -1.0, 1.0)

    # Calculate cosine distance
    cost_matrix = 1.0 - similarity_matrix
    return cost_matrix

# Note: For Mahalanobis distance used in original DeepSORT gating, you would need
# the Kalman Filter's state (x) and covariance (P), and the measurement (z)
# and measurement noise (R). It's more involved than typically needed if just
# following the simplified matching logic presented.