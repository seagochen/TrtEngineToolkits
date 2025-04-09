import time
from typing import Union, Dict, List

import numpy as np


class StatisticsCache:

    def __init__(self,
                 retention_time: int = 30,
                 enable_record: bool = False,
                 data_format: Dict[str, int] = None):
        """
        @brief 行为记录缓存类，用于记录目标的行为数据。
        @param retention_time 数据保留时间（秒）
        @param enable_record 是否启用记录
        @param data_format 数据格式，格式为 {"timestamp": 0, "oid": 1, "cx": 2, "cy": 3, "cz": 4, "orientation": 5, "action": 6}
        """
        self.retention_time = retention_time
        self.enable_record = enable_record
        self.data_format = data_format
        self.statistic_data = None  # numpy array

    def _remove_expired_records(self, current_timestamp: float):
        """
        @brief 移除过期记录，即时间戳小于当前时间减去数据保留时间的记录
        @param current_timestamp: 当前时间戳，小于该时间戳的记录将被移除
        @return: None
        """
        if self.statistic_data is None or self.statistic_data.size == 0:
            return

        # 获取时间戳所在列（按照 data_format 中的定义）
        t_pos = self.data_format.get("timestamp", 0)
        self.statistic_data = self.statistic_data[self.statistic_data[:, t_pos] > current_timestamp - self.retention_time]

    def _write_csv_records(self, data_record: Dict[str, Union[int, float]]):
        """
        @brief 将记录写入 CSV 文件，按照 data_format 定义的顺序写入
        @param data_record: 单条数据记录，格式为 {"timestamp": 0, "oid": 1, ...}
        @return: None
        """
        # 获取有序的键列表，按 data_format 中的值排序
        ordered_keys = sorted(self.data_format.keys(), key=lambda x: self.data_format[x])

        with open("statistics_data.csv", "a") as f:
            # 写入标题（如果文件为空）
            if f.tell() == 0:
                f.write(",".join(ordered_keys) + "\n")
            # 写入数据记录，按照有序键列表
            f.write(",".join([str(data_record.get(k, 0)) for k in ordered_keys]) + "\n")

    def add_entity(self, data_record: Dict[str, Union[int, float]]):
        """
        @brief 添加目标数据记录
        @param data_record: 数据记录，格式为 {"timestamp": 0, "oid": 1, "cx": 2, ...}
        @return: None
        """
        # 按照 data_format 排序键，确保数据顺序一致
        ordered_keys = sorted(self.data_format.keys(), key=lambda x: self.data_format[x])
        record_list = [data_record.get(key, 0) for key in ordered_keys]

        if self.statistic_data is None:
            self.statistic_data = np.array([record_list])
        else:
            self.statistic_data = np.vstack((self.statistic_data, record_list))

        # 使用当前记录的 timestamp（若无则使用当前时间）移除过期记录
        self._remove_expired_records(data_record.get("timestamp", time.time()))

        # 如果启用记录，则写入 CSV 文件
        if self.enable_record:
            self._write_csv_records(data_record)

    def count_data(self, id_key: str, id_val: int) -> int:
        """
        @brief 统计目标数量
        @param id_key: 数据格式中的字段名
        @param id_val: 要匹配的值
        @return: 匹配记录的数量
        """
        if self.statistic_data is None:
            return 0

        id_pos = self.data_format.get(id_key, 0)
        return len(self.statistic_data[self.statistic_data[:, id_pos] == id_val])

    def remove_data(self, id_key: str, id_val: int):
        """
        @brief 移除目标数据
        @param id_key: 数据格式中的字段名
        @param id_val: 要移除的值
        @return: None
        """
        if self.statistic_data is None:
            return

        id_pos = self.data_format.get(id_key, 0)
        self.statistic_data = self.statistic_data[self.statistic_data[:, id_pos] != id_val]

    def get_data(self, id_key: str, id_val: int) -> np.ndarray:
        """
        @brief 获取目标数据
        @param id_key: 数据格式中的字段名
        @param id_val: 要匹配的值
        @return: 匹配记录的 numpy 数组
        """
        if self.statistic_data is None:
            key_len = len(self.data_format)
            return np.zeros((0, key_len))

        id_pos = self.data_format.get(id_key, 0)
        return self.statistic_data[self.statistic_data[:, id_pos] == id_val]

    def get_unique_data(self, id_key: str) -> List[Union[int, float]]:
        """
        @brief 根据键获取唯一记录
        @param id_key: 数据格式中的字段名
        @return: 唯一记录列表
        """
        if self.statistic_data is None:
            key_len = len(self.data_format)
            return []

        id_pos = self.data_format.get(id_key, 0)
        return np.unique(self.statistic_data[:, id_pos]).tolist()

    def clear(self):
        """
        @brief 清空统计数据
        """
        self.statistic_data = None

    def has_data(self, id_key: str, id_val: int) -> bool:
        """
        @brief 判断是否有目标数据
        @param id_key: 数据格式中的字段名
        @param id_val: 要匹配的值
        @return: 是否存在目标数据
        """
        if self.statistic_data is None:
            return False

        id_pos = self.data_format.get(id_key, 0)
        return len(self.statistic_data[self.statistic_data[:, id_pos] == id_val]) > 0



def test_statistics_cache_basic():
    # 定义数据格式（按照所需顺序）
    data_format = {"timestamp": 0, "oid": 1, "cx": 2, "cy": 3, "cz": 4, "orientation": 5, "action": 6}
    cache = StatisticsCache(retention_time=30, enable_record=False, data_format=data_format)

    # 添加三条记录，其中 oid 为 1 的有两条，oid 为 2 的有一条
    now = time.time()
    record1 = {"timestamp": now, "oid": 1, "cx": 100.0, "cy": 200.0, "cz": 300.0, "orientation": 0, "action": 1}
    record2 = {"timestamp": now, "oid": 2, "cx": 110.0, "cy": 210.0, "cz": 310.0, "orientation": 1, "action": 2}
    record3 = {"timestamp": now, "oid": 1, "cx": 120.0, "cy": 220.0, "cz": 320.0, "orientation": 0, "action": 3}

    cache.add_entity(record1)
    cache.add_entity(record2)
    cache.add_entity(record3)

    print("After adding 3 records:")
    print(cache.statistic_data)

    # 统计 oid 为 1 的记录数量（预期为2）
    count_oid1 = cache.count_data("oid", 1)
    print("Count for oid=1:", count_oid1)

    # 统计 oid 为 2 的记录数量（预期为1）
    count_oid2 = cache.count_data("oid", 2)
    print("Count for oid=2:", count_oid2)

    # 删除 oid 为 2 的记录
    cache.remove_data("oid", 2)
    count_oid2_after = cache.count_data("oid", 2)
    print("Count for oid=2 after removal:", count_oid2_after)

    # 如果修改 get_data 接口为接收键和值，则可以测试：
    # data_oid1 = cache.get_data("oid", 1)
    # print("Data for oid=1:", data_oid1)


def test_get_data_fix():
    # 假设我们修改 get_data 接口如下：
    def get_data(self, id_key: str, id_val: Union[int, float, str]) -> np.ndarray:
        if self.statistic_data is None:
            key_len = len(self.data_format.keys())
            return np.zeros((0, key_len))
        id_pos = self.data_format.get(id_key, 0)
        return self.statistic_data[self.statistic_data[:, id_pos] == id_val]

    # 为测试临时绑定一个新方法
    StatisticsCache.get_data = get_data

    data_format = {"timestamp": 0, "oid": 1, "cx": 2, "cy": 3, "cz": 4, "orientation": 5, "action": 6}
    cache = StatisticsCache(retention_time=30, enable_record=False, data_format=data_format)

    now = time.time()
    record1 = {"timestamp": now, "oid": 1, "cx": 100.0, "cy": 200.0, "cz": 300.0, "orientation": 0, "action": 1}
    record2 = {"timestamp": now, "oid": 1, "cx": 105.0, "cy": 205.0, "cz": 305.0, "orientation": 0, "action": 2}
    record3 = {"timestamp": now, "oid": 2, "cx": 110.0, "cy": 210.0, "cz": 310.0, "orientation": 1, "action": 3}

    cache.add_entity(record1)
    cache.add_entity(record2)
    cache.add_entity(record3)

    data_oid1 = cache.get_data("oid", 1)
    print("Data for oid=1 (should have 2 rows):")
    print(data_oid1)


if __name__ == "__main__":
    print("=== Running basic test ===")
    test_statistics_cache_basic()
    print("\n=== Running get_data fix test ===")
    test_get_data_fix()
