import logging
import sys
import traceback
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
        # 通过 __new__ 方法实现单例模式，确保无论在哪里调用 Logger，都返回同一个实例。
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
            # 格式化日志消息
            formatted_message = f"{log_color}[{record.levelname}] <{timestamp}> {record.msg}{RESET}"
            # 如果有异常信息，则添加调用栈信息
            if record.exc_info:
                formatted_message += "\n" + self.formatException(record.exc_info)
            return formatted_message

    def log(self, level, module, message, topic=None, exc_info=False):
        # 拼接模块名和 topic 信息
        log_message = f"[{module}]"
        if topic:
            log_message += f"/{topic}"
        log_message += f" - {message}"

        if level == "VERBOSE":
            self.logger.log(VERBOSE_LEVEL, log_message, exc_info=exc_info)
        elif level == "INFO":
            self.logger.info(log_message, exc_info=exc_info)
        elif level == "WARNING":
            self.logger.warning(log_message, exc_info=exc_info)
        elif level == "ERROR":
            self.logger.error(log_message, exc_info=exc_info)
        elif level == "DEBUG":
            self.logger.debug(log_message, exc_info=exc_info)
        elif level == "CRITICAL":
            self.logger.critical(log_message, exc_info=exc_info)

    def verbose(self, module, message):
        self.log("VERBOSE", module, message)

    def info(self, module, message):
        self.log("INFO", module, message)

    def warning(self, module, message):
        self.log("WARNING", module, message)

    def error(self, module, message):
        # 默认 error 不附带调用栈
        self.log("ERROR", module, message)

    def error_trace(self, module, message):
        """
        记录错误信息并附带异常调用栈信息。
        在 except 块中调用此方法，即可输出详细的调用栈信息。
        """
        self.log("ERROR", module, message, exc_info=True)

    def debug(self, module, message):
        self.log("DEBUG", module, message)

    def critical(self, module, message):
        self.log("CRITICAL", module, message)


# 定义一个 logger 实例
logger = Logger()

# 示例用法
if __name__ == "__main__":
    try:
        1 / 0
    except Exception as e:
        logger.error_trace("ModuleA", f"发生错误: {e}")
    logger.verbose("ModuleA", "This is a verbose message.")
    logger.info("ModuleA", "This is an info message.")
    logger.critical("ModuleB", "This is a critical message.")
    logger.warning("ModuleB", "This is a warning message.")
    logger.error("ModuleC", "This is an error message without stack trace.")
    logger.debug("ModuleC", "This is a debug message.")
