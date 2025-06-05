
from dataclasses import dataclass
from typing import Tuple


# -------------- 定义 dataclass --------------
@dataclass
class Posture:
    """
    人体姿态信息：
      action: 动作编码
          0 - 未知
          1 - 弯腰
          2 - 坐
          3 - 下蹲
          4 - 站立
    """
    action: int

@dataclass
class FacialDirection:
    """
    面部朝向信息：
      modulus: 模长，根据检测框宽度计算得到
      vector: 单位方向向量 (vec_x, vec_y)
      origin: 原点坐标 (origin_x, origin_y)
      direction_desc: 方向描述 ("Front", "Left", "Right", "Back", "Unknown")
      direction_type: 离散方向编码（0: 正面, 1: 左侧, 2: 右侧, 3: 背面, -1: 未知）
    """
    modulus: int
    vector: Tuple[float, float]
    origin: Tuple[int, int]
    direction_desc: str
    direction_type: int