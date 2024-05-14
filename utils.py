import os
import logging
import sys
from logging import handlers
from os.path import join

__all__ = ['get_logger', 'get_project_path', 'project_path', 'get_default_path']


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

print(f"current project path：{project_path}")


def get_default_path(dataset_name):
    assert dataset_name in ["tju_pv600", "hkpu_pv500", "vera_pv220"]
    evolution_output_dir_default = join(project_path, "ckpts", "evolution", f"{dataset_name}")
    evolution_checkpoint_path_default = None
    model_cfg_path_default = join(project_path, 'models', 'configs', 'vim', f"{dataset_name}.yaml")
    model_save_path_default = join(project_path, 'ckpts', 'models', f"{dataset_name}")
    return (evolution_output_dir_default, evolution_checkpoint_path_default,
            model_cfg_path_default, model_save_path_default)



