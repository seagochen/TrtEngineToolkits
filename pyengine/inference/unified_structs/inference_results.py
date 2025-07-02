# pyengine/inference/unified_structs/inference_results.py

import json
from dataclasses import dataclass, field, asdict, astuple
from typing import List, Union, Any, Type, TypeVar, Tuple

# 为泛型类方法定义 TypeVar
T = TypeVar('T', bound='InferenceResult')


@dataclass
class InferenceResult:
    """基类，提供通用的序列化和反序列化方法。"""

    def to_list(self) -> Tuple[Any, ...]:
        """将数据类实例转换为其字段值的元组。"""
        return astuple(self)

    def to_dict(self: T) -> dict[str, Any]:
        """将数据类实例转换为字段名和值的字典。"""
        return asdict(self)

    def to_json(self, indent: int = 4) -> str:
        """将数据类实例转换为 JSON 字符串。"""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_list(cls: Type[T], data: List[Any]) -> T:
        """从值列表创建数据类实例。"""
        return cls(*data)

    @classmethod
    def from_dict(cls: Type[T], data: dict[str, Any]) -> T:
        """从字典创建数据类实例。"""
        return cls(**data)

    @classmethod
    def from_json(cls: Type[T], data: str) -> Union[T, List[T]]:
        """从 JSON 字符串创建一个或多个数据类实例。"""
        try:
            data_parsed = json.loads(data)
        except json.JSONDecodeError as e:
            raise ValueError("无效的 JSON 数据") from e

        if isinstance(data_parsed, list):
            if not all(isinstance(item, dict) for item in data_parsed):
                raise TypeError("JSON 列表必须只包含字典")
            return [cls.from_dict(item) for item in data_parsed]
        elif isinstance(data_parsed, dict):
            return cls.from_dict(data_parsed)
        else:
            raise TypeError("JSON 必须表示一个字典或字典列表")


@dataclass
class Rect:
    """Represents a rectangle with its top-left and bottom-right coordinates."""
    x1: float = 0.0
    y1: float = 0.0
    x2: float = 0.0
    y2: float = 0.0


@dataclass
class ObjectDetection(InferenceResult):
    """Represents detection bounding box with class and confidence."""
    rect: Rect = field(default_factory=Rect)
    classification: int = 0
    confidence: float = 0.0
    track_id: int = 0
    features: List[float] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ObjectDetection":
        """从字典创建实例，并手动将嵌套的 'rect' 字典转换为 Rect 对象。"""
        if 'rect' in data and isinstance(data.get('rect'), dict):
            data['rect'] = Rect(**data['rect'])
        return super().from_dict(data)


@dataclass
class Point(InferenceResult):
    """Represents a single keypoint with its coordinates and confidence."""
    x: float = 0.0
    y: float = 0.0
    confidence: float = 0.0


@dataclass
class Skeleton(ObjectDetection):
    """Represents a human skeleton, inheriting bounding box info and adding keypoints."""
    points: List[Point] = field(default_factory=list)

    # ##################################################################
    # #                           修改部分 START                          #
    # ##################################################################
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Skeleton":
        """
        从字典创建实例。
        仅处理本类定义的嵌套对象 'points'，然后将其他字段交由父类处理。
        """
        # 1. (优化) 只处理本层级定义的 'points' 字段
        if 'points' in data and isinstance(data.get('points'), list):
            # 使用 Point.from_dict 而不是 Point(**p) 是更稳健的做法
            data['points'] = [Point.from_dict(p) for p in data['points']]

        # 2. (删除) 移除对 'rect' 的重复处理，将其完全交给父类
        # if 'rect' in data and isinstance(data.get('rect'), dict):
        #     data['rect'] = Rect(**data['rect'])

        # 3. (不变) 调用父类的 from_dict，让它去处理 'rect' 等字段
        return super().from_dict(data)
    # ##################################################################
    # #                           修改部分 END                            #
    # ##################################################################