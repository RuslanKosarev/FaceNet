# flake8: noqa
# coding:utf-8

from tensorflow.python.framework import dtypes
from facenet import tfutils, ioutils

nodes = {
    'input': {
        'name': 'input',
        'type': dtypes.uint8.as_datatype_enum
        },

    'output': {
        'name': 'embeddings',
        'type': dtypes.float32.as_datatype_enum
    },

}

config_nodes = {
    'image_size': {
        'name': 'image_size:0',
        'type': dtypes.uint8.as_datatype_enum
    }
}
