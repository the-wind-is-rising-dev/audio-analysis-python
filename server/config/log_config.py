import logging
import os
import sys
import threading
import time
import traceback


class LogConfig:

    def __init__(self, name: any, log_type: str, log_file: str):
        self.log_file = log_file
        self.log_type = log_type
        self.max_size = 100 * 1024 * 1024
        self.trim_size = 99 * 1024 * 1024
        self._last_check = time.time()
        # 为 LogWriter 创建专用的日志记录器
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        # 避免重复添加处理器
        if not self.logger.handlers:
            handler = logging.FileHandler(log_file)
            # %(asctime)s: 时间戳
            # %(message)s: 日志消息
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def check_and_trim_file(self):
        """检查文件大小并在必要时截断"""
        if os.path.exists(self.log_file):
            file_size = os.path.getsize(self.log_file)
            if file_size > self.max_size:
                try:
                    with open(self.log_file, 'rb') as f:
                        f.seek(self.trim_size)
                        remaining_data = f.read()

                    with open(self.log_file, 'wb') as f:
                        f.write(remaining_data)
                except Exception as e:
                    # 如果截断失败，继续正常写入
                    pass

    def write(self, message):
        if message.strip():  # 忽略空行
            # 减少文件大小检查频率
            current_time = time.time()
            if (current_time - self._last_check) > 30:
                # 检查是否需要截断文件
                self.check_and_trim_file()
                self._last_check = current_time

            # 创建日志记录器并手动记录
            thread_name = threading.current_thread().name
            formatted_message = f"{thread_name} - {message.strip()}"
            self.logger.info(formatted_message)
            # 获取当前时间戳
            timestamp = time.time()
            # 转换为本地时间结构
            time_struct = time.localtime(timestamp)
            # 格式化时间
            formatted_time = time.strftime('%Y-%m-%d %H:%M:%S', time_struct)
            if 'err' in self.log_type:
                sys.__stderr__.write(f'{formatted_time} - {formatted_message}\n')
            else:
                sys.__stdout__.write(f'{formatted_time} - {formatted_message}\n')

    def flush(self):
        if 'err' in self.log_type:
            sys.__stderr__.flush()
        else:
            sys.__stdout__.flush()


def log_config(name: any, log_filename: str):
    """配置日志"""
    try:
        # 创建日志写入器=
        # 重定向 print 输出
        sys.stdout = LogConfig(name=name, log_type='out', log_file=log_filename)
        sys.stderr = LogConfig(name=name, log_type='err', log_file=log_filename)
    except Exception as e:
        traceback.print_exc()
        # 如果重定向失败，至少保证程序能正常运行
        print(f"日志重定向配置失败: {e}")
