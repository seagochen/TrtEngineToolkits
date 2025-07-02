# auxiliary_structs.py

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

from pyengine.inference.unified_structs.inference_results import Skeleton


class FaceDirection(IntEnum):
    Front = 0
    Left = 1
    Right = 2
    Back = 3
    Unknown = -1

    def __str__(self): return self.name.lower()

    @classmethod
    def from_value(cls, value): return cls(value)


class Pose(IntEnum):
    Standing = 0
    Bending = 1
    Sitting = 2
    Unknown = -1

    def __str__(self): return self.name.lower()

    @classmethod
    def from_value(cls, value): return cls(value)


@dataclass
class ExtendedSkeleton(Skeleton):

    # 用于标记姿态
    pose: Pose = Pose.Unknown

    # ##################################################################
    # #                           修改部分 START                          #
    # ##################################################################

    # (修改) 修正拼写错误 directionn_type -> direction
    direction: FaceDirection = FaceDirection.Unknown

    # 关于面部朝向的其他辅助信息，例如角度，向量，模长
    direction_angle: float = 0.0
    direction_modulus: float = 0.0
    direction_vector: tuple[float, float] = (0.0, 0.0)
    direction_origin: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self):
        """
        在对象初始化后，确保 direction 和 pose 是正确的枚举类型。
        """
        if hasattr(super(), '__post_init__'):
            super().__post_init__()

        # (修改) 使用修正后的字段名 self.direction
        if not isinstance(self.direction, FaceDirection):
            self.direction = FaceDirection.from_value(self.direction)
        if not isinstance(self.pose, Pose):
            self.pose = Pose.from_value(self.pose)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExtendedSkeleton":
        """
        (新增) 从字典创建实例。
        这个方法是必需的，以确保反序列化逻辑链的完整性。
        它调用父类的方法来处理所有嵌套对象的转换，
        同时确保最终的实例是作为 ExtendedSkeleton 创建的，
        从而能够正确接收 pose 和 direction 等新字段。
        """
        # 调用父类的 from_dict，它会处理 'points' 和 'rect' 的转换
        return super().from_dict(data)
    # ##################################################################
    # #                           修改部分 END                            #
    # ##################################################################