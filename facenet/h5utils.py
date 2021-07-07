# coding:utf-8
__author__ = 'Ruslan N. Kosarev'

import numpy as np
import h5py
from pathlib import Path


def write(file, name, data, mode='a'):
    file = Path(file).expanduser()
    name = str(name)
    data = np.atleast_1d(data)

    with h5py.File(file, mode=mode) as hf:
        if name in hf:
            del hf[name]
        hf.create_dataset(name, data=data, compression='gzip', dtype=data.dtype)


def read(file, name, default=None):
    with h5py.File(str(file), mode='r') as hf:
        if name in hf:
            return hf[name][...]
        else:
            if default is not None:
                return default
            else:
                raise KeyError(f'Invalid key {name} in H5 file {file}')
