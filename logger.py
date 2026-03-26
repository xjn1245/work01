"""
日志配置模块

该模块配置了项目的日志记录器，用于记录运行过程中的各种信息、警告和错误。
"""

import logging

# 配置日志器
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # 设置日志级别为DEBUG，捕获所有级别的日志

# 配置日志格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 配置文件处理器，将日志写入文件
fhandler = logging.FileHandler('livetalking.log')  # 可以改为StreamHandler输出到控制台或多个Handler组合使用等。
fhandler.setFormatter(formatter)
fhandler.setLevel(logging.INFO)  # 文件处理器设置为INFO级别，只记录INFO及以上级别的日志
logger.addHandler(fhandler)

# 配置控制台处理器，将日志输出到控制台
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
sformatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(sformatter)
logger.addHandler(handler)
