import numpy as np
from scipy.signal import savgol_filter, butter, filtfilt


def moving_average(data, window_size=5):
    """
    :param data: np.ndarray, 输入的坐标数据 [n_samples, 3] (x, y, z)
    :param window_size: int, 滑动窗口的大小
    :return: np.ndarray, 平滑后的坐标数据
    """
    if len(data) < window_size:
        return data  # 数据点不足窗口大小时直接返回原数据

    smoothed_data = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    # 补齐前后端的点数，使得输出长度与输入一致
    pad = (len(data) - len(smoothed_data)) // 2
    smoothed_data = np.pad(smoothed_data, (pad, pad), mode='edge')
    return smoothed_data


"""
smoothed_x = moving_average(data[:, 0])
smoothed_y = moving_average(data[:, 1])
smoothed_z = moving_average(data[:, 2])
"""

def exponential_moving_average(data, alpha=0.1):
    """
    :param data: np.ndarray, 输入的坐标数据 [n_samples]
    :param alpha: float, 平滑系数 (0, 1)，越接近1越灵敏
    :return: np.ndarray, 平滑后的数据
    """
    ema = np.zeros_like(data)
    ema[0] = data[0]
    for t in range(1, len(data)):
        ema[t] = alpha * data[t] + (1 - alpha) * ema[t-1]
    return ema


"""
smoothed_x = exponential_moving_average(data[:, 0])
smoothed_y = exponential_moving_average(data[:, 1])
smoothed_z = exponential_moving_average(data[:, 2])
"""


def savitzky_golay(data, window_size=5, poly_order=2):
    """
    :param data: np.ndarray, 输入的坐标数据 [n_samples]
    :param window_size: int, 滑动窗口大小，必须为奇数
    :param poly_order: int, 拟合多项式的阶数
    :return: np.ndarray, 平滑后的数据
    """
    if len(data) < window_size:
        return data  # 数据点不足窗口大小时直接返回原数据

    return savgol_filter(data, window_size, poly_order)


"""
smoothed_x = savitzky_golay(data[:, 0])
smoothed_y = savitzky_golay(data[:, 1])
smoothed_z = savitzky_golay(data[:, 2])
"""


def low_pass_filter(data, cutoff=0.1, fs=1.0, order=3):
    """
    :param data: np.ndarray, 输入的坐标数据 [n_samples]
    :param cutoff: float, 截止频率 (0, 0.5)，越低抑制越多高频
    :param fs: float, 采样频率
    :param order: int, 滤波器的阶数
    :return: np.ndarray, 平滑后的数据
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)


"""
smoothed_x = low_pass_filter(data[:, 0])
smoothed_y = low_pass_filter(data[:, 1])
smoothed_z = low_pass_filter(data[:, 2])
"""
