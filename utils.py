import os
import logging
import sys
from logging import handlers


def get_logger(file_name, level=logging.INFO, mode="a", format="%(asctime)s %(levelname)s: %(message)s",
               datamat="%Y-%m-%d %H:%M:%S", maxBytes=1024000, backCount=5):
    logger = logging.getLogger()
    logger.setLevel(level)
    formater = logging.Formatter(fmt=format, datefmt=datamat)

    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(fmt=formater)
    th = handlers.RotatingFileHandler(filename=file_name, maxBytes=maxBytes, mode=mode,
                                      backupCount=backCount, encoding='utf-8')
    th.setFormatter(fmt=formater)
    logger.addHandler(sh)
    logger.addHandler(th)
    return logger


def get_project_path(project_name):
    """
    :param project_name: 项目名称，如pythonProject
    :return: ******/project_name
    """
    # 获取当前所在文件的路径
    cur_path = os.path.abspath(os.path.dirname(__file__))

    # 获取根目录
    return cur_path[:cur_path.find(project_name)] + project_name


project_path = get_project_path(project_name="MambaNAS")



