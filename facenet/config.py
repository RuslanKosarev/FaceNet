# coding: utf-8
__author__ = 'Ruslan N. Kosarev'

import yaml
import pathlib

src_dir = pathlib.Path(__file__).parents[1]
file_extension = '.png'


def default_app_config(apps_file_name):
    config_dir = pathlib.Path(pathlib.Path(apps_file_name).parent).joinpath('configs')
    config_name = pathlib.Path(apps_file_name).stem
    return config_dir.joinpath(config_name + '.yaml')


def replace_str_value(x):
    dct = {'none': None, 'false': False, 'true': True}

    if isinstance(x, str):
        for name, value in dct.items():
            if x.lower() == name:
                return value
    return x


# class Namespace:
#     """Simple object for storing attributes.
#     Implements equality by attribute names and values, and provides a simple string representation.
#     """
#
#     def __init__(self, dct):
#         for key, item in dct.items():
#             if isinstance(item, dict):
#                 setattr(self, key, Namespace(item))
#             else:
#                 setattr(self, key, replace_str_value(item))
#
#     def items(self):
#         return self.__dict__.items()
#
#     def __repr__(self):
#         return "<namespace object>"


class YAMLConfig:
    """Object representing YAML settings as a dict-like object with values as fields
    """

    def __init__(self, item):
        if isinstance(item, dict):
            self.update(item)
        else:
            self.update_from_file(item)

    def update(self, dct):
        """Update config from dict

        :param dct: dict
        """
        for key, item in dct.items():
            if isinstance(item, dict):
                setattr(self, key, YAMLConfig(item))
            else:
                setattr(self, key, replace_str_value(item))

    def update_from_file(self, path):
        """Update config from YAML file
        """
        if path is not None:
            with open(str(path), 'r') as custom_config:
                self.update(yaml.safe_load(custom_config.read()))

    def items(self):
        return self.__dict__.items()

    def __repr__(self):
        return "<config object>"


class DefaultConfig:
    def __init__(self):
        self.model = src_dir.joinpath('models', '20190727-080213')
        self.pretrained_checkpoint = src_dir.joinpath('models', '20190727-080213', 'model-20190727-080213.ckpt-275')

        # type of distance metric to use. 0: Euclidian, 1:Cosine similarity distance
        self.distance_metric = 0

        # image size (height, width) in pixels
        self.image_size = 160

        # embedding size
        self.embedding_size = 512

        # image standardisation
        # False: tf.image.per_image_standardization(image)
        # True: (tf.cast(image, tf.float32) - 127.5) / 128.0
        self.image_standardization = True
