# coding:utf-8

from loguru import logger


def configure_logging(cfg):
    """Configure the application logging
    """
    file = cfg.dir / cfg.file

    logger.add(file)



