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
WHITE = "\033[97m"

# 日志级别的颜色映射
LOG_LEVEL_COLORS = {
    "VERBOSE": CYAN,
    "INFO": GREEN,
    "WARNING": YELLOW,
    "ERROR": RED,
    "DEBUG": GRAY,
    "CRITICAL": WHITE
}

# 自定义日志级别 VERBOSE
VERBOSE_LEVEL = 15
logging.addLevelName(VERBOSE_LEVEL, "VERBOSE")


class Logger:
    _instance = None

    def __new__(cls, *args, **kwargs):
        # 通过new 通过 __new__ 方法实现了单例模式，确保无论在哪里调用 Logger，都返回同一个实例。
        # __new__ 方法： __new__ 是创建实例的一个特殊方法，它会在 __init__ 方法之前执行。
        # 通过重写 __new__，我们可以控制实例的创建流程，确保只创建一次实例。

        if not cls._instance:
            cls._instance = super(Logger, cls).__new__(cls, *args, **kwargs)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        # 配置基本的日志记录器
        self.logger = logging.getLogger("Logger")
        self.logger.setLevel(logging.DEBUG)  # 设置最低日志级别

        if not self.logger.hasHandlers():
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(self.CustomFormatter())
            self.logger.addHandler(handler)

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            log_color = LOG_LEVEL_COLORS.get(record.levelname, RESET)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            level = record.levelname

            formatted_message = f"{log_color}[{level}] {record.levelname} <{timestamp}> {record.msg}{RESET}"
            return formatted_message

    def log(self, level, module, message, topic=None):
        log_message = f"[{module}]"
        if topic:
            log_message += f"/{topic}"
        log_message += f" - {message}"

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
        elif level == "CRITICAL":
            self.logger.critical(log_message)

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

    def critical(self, module, message):
        self.log("CRITICAL", module, message)


# 示例用法
if __name__ == "__main__":
    logger = Logger()

    logger.verbose("ModuleA", "This is a verbose message.")
    logger.info("ModuleA", "This is an info message.")
    logger.critical("ModelB", "This is a critical message.")
    logger.warning("ModuleB", "This is a warning message.")
    logger.error("ModuleC", "This is an error message.")
    logger.debug("ModuleC", "This is a debug message.")
