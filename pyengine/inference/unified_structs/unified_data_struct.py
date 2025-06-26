import json
from dataclasses import dataclass, field, asdict, astuple
from typing import List, Union, Any, Type, TypeVar, Tuple

# 为泛型类方法定义 TypeVar
T = TypeVar('T', bound='YoloBase')


@dataclass
class YoloBase:
    """基类，提供通用的序列化和反序列化方法。"""

    def to_list(self) -> Tuple[Any, ...]:
        """将数据类实例转换为其字段值的元组。"""
        return astuple(self)

    def to_dict(self) -> dict[str, Any]:
        """将数据类实例转换为字段名和值的字典。"""
        return asdict(self)

    def to_json(self, indent: int = 4) -> str:
        """将数据类实例转换为 JSON 字符串。"""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_list(cls: Type[T], data: List[Any]) -> T:
        """从值列表创建数据类实例。
        注意：对于字段顺序与列表顺序一致的简单数据类，此方法效果最佳。
        """
        # 类型检查器可能会对 *data 警告，但在运行时，只要 data 与字段匹配就是正确的。
        return cls(*data)

    @classmethod
    def from_dict(cls: Type[T], data: dict[str, Any]) -> T:
        """从字典创建数据类实例。
        注意：对于字典键与字段名一致的简单数据类，此方法效果最佳。
        """
        # 类型检查器可能会对 **data 警告，但在运行时，只要 data 与字段匹配就是正确的。
        return cls(**data)

    @classmethod
    def from_json(cls: Type[T], data: str) -> Union[T, List[T]]:
        """从 JSON 字符串创建一个或多个数据类实例。"""
        try:
            data_parsed = json.loads(data)
        except json.JSONDecodeError as e:
            raise ValueError("无效的 JSON 数据") from e

        if isinstance(data_parsed, list):
            # 确保列表中的每一项都是字典，以便 from_dict 使用
            if not all(isinstance(item, dict) for item in data_parsed):
                raise TypeError("JSON 列表必须只包含字典")
            return [cls.from_dict(item) for item in data_parsed]
        elif isinstance(data_parsed, dict):
            return cls.from_dict(data_parsed)
        else:
            raise TypeError("JSON 必须表示一个字典或字典列表")
