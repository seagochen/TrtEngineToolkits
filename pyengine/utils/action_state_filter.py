import time
from typing import Optional, Dict, Tuple, Union


class ActionStateFilter:
    """
    过滤重复状态，只在状态发生更新时返回信息，并对记录设置驻留时间，
    超过该时间后自动删除记录，避免长时间运行时内存占用过多。
    """
    def __init__(self, msg_retention_time: float = 60.0):
        """
        :param retention_time: 消息的驻留时间（秒），超过此时间后记录会被清理
        """
        self.msg_retention_time = msg_retention_time
        # 保存每个目标的最新状态，结构为： {oid: (message, timestamp)}
        self.last_states: Dict[Union[str, int], Tuple[str, float]] = {}

    def cleanup(self, current_timestamp: float):
        """
        清理超过驻留时间的记录
        :param current_timestamp: 当前时间戳
        """
        expired_oids = [
            oid for oid, (_, ts) in self.last_states.items()
            if current_timestamp - ts > self.msg_retention_time
        ]
        for oid in expired_oids:
            del self.last_states[oid]

    def process_message(self, oid: Union[str, int], message: str, timestamp: float) -> Optional[str]:
        """
        处理一条消息，只在状态更新时返回消息，否则返回 None

        :param oid: 目标ID
        :param message: AI 模型生成的检测信息
        :param timestamp: 消息生成的时间戳（秒）
        :return: 状态更新时返回消息；如果状态未更新则返回 None
        """
        # 清理过期记录
        self.cleanup(timestamp)

        # 如果该目标没有记录，则直接保存并返回消息
        if oid not in self.last_states:
            self.last_states[oid] = (message, timestamp)
            return message
        else:
            stored_message, stored_ts = self.last_states[oid]
            if message == stored_message:
                # 状态未更新
                return None
            else:
                # 状态更新，保存新状态并返回消息
                self.last_states[oid] = (message, timestamp)
                return message

# 示例测试代码
if __name__ == "__main__":
    filter = ActionStateFilter(msg_retention_time=5.0)  # 比如设置5秒的驻留时间
    test_data = [
        ("目标1", "正在刷牙", 10.001),
        ("目标1", "正在刷牙", 10.005),
        ("目标2", "正在刷牙", 10.007),
        ("目标3", "正在刷牙", 10.008),
        ("目标1", "正在刷牙", 10.013),
        ("目标1", "刷牙结束", 11.012),
        ("目标2", "正在刷牙", 11.017),
    ]

    for oid, msg, ts in test_data:
        result = filter.process_message(oid, msg, ts)
        if result is not None:
            print(f"{oid} {result} at {ts}")
