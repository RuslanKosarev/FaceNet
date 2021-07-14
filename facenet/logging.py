# coding:utf-8

from typing import Union
from pathlib import Path
from loguru import logger


def configure_logging(path: Union[Path, str], file_name: str='log.txt'):
    """Configure the application logging
    """
    logger.add(Path(path) / file_name)




