import time

class Blinker:

    def __init__(self, blink_interval=0.2):
        """
        :param blink_interval: 闪烁状态切换的时间间隔，单位为秒。
        """
        self.last_blink_time = time.time()
        self.blink_on = True
        self.blink_interval = blink_interval

    def state_changed(self):
        """
        每次调用时检查是否应该根据时间间隔切换状态。

        :return: 返回当前的闪烁状态（True 或 False）。
        """
        current_time = time.time()
        elapsed_time = current_time - self.last_blink_time
        
        # 如果经过的时间超过设定的间隔时间，翻转状态
        if elapsed_time >= self.blink_interval:
            # 计算状态翻转的次数
            flip_count = int(elapsed_time // self.blink_interval)
            
            # 状态翻转flip_count次，如果是奇数次则翻转状态
            if flip_count % 2 != 0:
                self.blink_on = not self.blink_on

            # 更新上次状态翻转的时间，保留余下的时间部分
            self.last_blink_time = current_time - (elapsed_time % self.blink_interval)

        return self.blink_on


if __name__ == "__main__":
    blinker = Blinker(0.5)  # 每0.5秒翻转状态

    for i in range(20):
        time.sleep(0.1)  # 模拟每0.1秒调用一次
        print(f"{i}: Blinker state is {'ON' if blinker.state_changed() else 'OFF'}")
