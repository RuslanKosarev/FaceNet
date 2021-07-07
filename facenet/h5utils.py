# coding:utf-8
__author__ = 'Ruslan N. Kosarev'

from typing import Union
import h5py
from pathlib import Path


def write(file: Union[Path, str], path: str, data, mode='a'):

    file = Path(file).expanduser()

    with h5py.File(file, mode=mode) as hf:
        if path in hf:
            del hf[path]
        hf[path] = data


def read(file: Union[Path, str], path: str, **kwargs):

    file = Path(file).expanduser()

    with h5py.File(file, mode='r') as hf:
        if path in hf:
            return hf[path][...]

        if 'default' in kwargs.keys():
            return kwargs['default']
        else:
            raise KeyError(f'Invalid path {path} in h5 file {file}')
