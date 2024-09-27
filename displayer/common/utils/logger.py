import logging
import sys
from datetime import datetime

# ANSI 转义码定义颜色
RESET = "\033[0m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
GRAY = "\033[90m"

# 日志级别的颜色映射
LOG_LEVEL_COLORS = {
    "VERBOSE": CYAN,
    "INFO": GREEN,
    "WARNING": YELLOW,
    "ERROR": RED,
    "DEBUG": GRAY,
}

# 自定义日志级别 VERBOSE
VERBOSE_LEVEL = 15
logging.addLevelName(VERBOSE_LEVEL, "VERBOSE")

class Logger:
    def __init__(self):
        # 配置基本的日志记录器
        self.logger = logging.getLogger("Logger")
        self.logger.setLevel(logging.DEBUG)  # 设置最低日志级别

        # Check if the logger already has handlers to prevent duplicate logging
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(self.CustomFormatter())
            self.logger.addHandler(handler)

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            log_color = LOG_LEVEL_COLORS.get(record.levelname, RESET)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            level = record.levelname  # 使用日志级别名称替代 Logger

            # 格式化日志消息
            formatted_message = f"{log_color}[{level}] {record.levelname} {timestamp}: {record.msg}{RESET}"
            return formatted_message

    def log(self, level, module, message, topic=None):
        log_message = f"{module}"
        if topic:
            log_message += f"/{topic}"
        log_message += f": {message}"

        if level == "VERBOSE":
            self.logger.log(VERBOSE_LEVEL, log_message)
        elif level == "INFO":
            self.logger.info(log_message)
        elif level == "WARNING":
            self.logger.warning(log_message)
        elif level == "ERROR":
            self.logger.error(log_message)
        elif level == "DEBUG":
            self.logger.debug(log_message)

    # 宏的 Python 实现等效
    def verbose(self, module, message):
        self.log("VERBOSE", module, message)

    def info(self, module, message):
        self.log("INFO", module, message)

    def warning(self, module, message):
        self.log("WARNING", module, message)

    def error(self, module, message):
        self.log("ERROR", module, message)

    def debug(self, module, message):
        self.log("DEBUG", module, message)

# 使用示例
if __name__ == "__main__":
    logger = Logger()

    logger.verbose("ModuleA", "This is a verbose message.")
    logger.info("ModuleA", "This is an info message.")
    logger.warning("ModuleA", "This is a warning message.")
    logger.error("ModuleA", "This is an error message.")
    logger.debug("ModuleA", "This is a debug message.")

    # 带 topic 的日志
    logger.log("INFO", "ModuleB", "This is a message with topic", topic="Topic1")
