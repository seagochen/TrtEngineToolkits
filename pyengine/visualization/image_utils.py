import numpy as np


def crop_grid_tile(frame: np.ndarray, rows:int=3, cols:int=3, index:int=5):
    """按 rows×cols 网格裁剪第 index 个格子（0 基）。"""
    h, w = frame.shape[:2]
    r, c = divmod(index, cols)
    cell_h = h // rows
    cell_w = w // cols

    y0 = r * cell_h
    y1 = (r + 1) * cell_h if r < rows - 1 else h
    x0 = c * cell_w
    x1 = (c + 1) * cell_w if c < cols - 1 else w
    return frame[y0:y1, x0:x1]