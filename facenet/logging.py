# coding:utf-8

from typing import Union
from pathlib import Path
from loguru import logger


def configure_logging(file: Union[Path, str]):
    """Configure the application logging
    """
    logger.add(file)




