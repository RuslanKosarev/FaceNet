# coding: utf-8
__author__ = 'Ruslan N. Kosarev'

import sys
import yaml
from pathlib import Path
from datetime import datetime
import importlib
import numpy as np
import random
from facenet import ioutils

src_dir = Path(__file__).parents[1]
file_extension = '.png'


def subdir():
    return datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')


def default_app_config(apps_file_name):
    config_dir = Path(Path(apps_file_name).parent).joinpath('configs')
    config_name = Path(apps_file_name).stem
    return config_dir.joinpath(config_name + '.yaml')


class YAMLConfig:
    """Object representing YAML settings as a dict-like object with values as fields
    """

    def __init__(self, item):
        if isinstance(item, dict):
            self.update_from_dict(item)
        else:
            self.update_from_file(item)

    def __repr__(self):
        shift = 3 * ' '

        def get_str(obj, ident=''):
            s = ''
            for key, item in obj.items():
                if isinstance(item, YAMLConfig):
                    s += '{}{}: \n{}'.format(ident, key, get_str(item, ident=ident + shift))
                else:
                    s += '{}{}: {}\n'.format(ident, key, str(item))
            return s

        return get_str(self)

    def __getattr__(self, name):
        return self.__dict__.get(name, YAMLConfig({}))

    def __bool__(self):
        return bool(self.__dict__)

    @staticmethod
    def check_item(s):
        if not isinstance(s, str):
            return s
        if s.lower() == 'none':
            return None
        if s.lower() == 'false':
            return False
        if s.lower() == 'true':
            return True
        return s

    def update_from_dict(self, dct):
        """Update config from dict

        :param dct: dict
        """
        for key, item in dct.items():
            if isinstance(item, dict):
                setattr(self, key, YAMLConfig(item))
            else:
                setattr(self, key, self.check_item(item))

    def update_from_file(self, path):
        """Update config from YAML file
        """
        if not path.exists():
            raise ValueError('file {} does not exist'.format(path))

        with path.open('r') as f:
            self.update_from_dict(yaml.safe_load(f.read()))

    def items(self):
        return self.__dict__.items()

    def exists(self, name):
        return True if name in self.__dict__.keys() else False


class Validate(YAMLConfig):
    def __init__(self, args_):
        YAMLConfig.__init__(self, args_['config'])
        if not self.seed:
            self.seed = 0
        random.seed(self.seed)
        np.random.seed(self.seed)

        if not self.model:
            self.model = DefaultConfig().model

        if not self.file:
            self.file = Path(self.model).expanduser().joinpath('report.txt')
        else:
            self.file = Path(self.file).expanduser()

        if not self.batch_size:
            self.batch_size = DefaultConfig().batch_size

        # write arguments and store some git revision info in a text files in the log directory
        ioutils.write_arguments(self, self.file.parent)
        ioutils.store_revision_info(self.file, sys.argv)


class TrainOptions(YAMLConfig):
    def __init__(self, args_, subdir=None):
        YAMLConfig.__init__(self, args_['config'])

        if not self.seed:
            self.seed = 0
        random.seed(self.seed)
        np.random.seed(self.seed)

        if subdir is None:
            self.model.path = Path(self.model.path).expanduser()
        else:
            self.model.path = Path(self.model.path).expanduser().joinpath(subdir)

        self.logs = self.model.path.joinpath('logs')
        self.h5file = self.logs.joinpath('statistics.h5')

        if self.model.config is None:
            network = importlib.import_module(self.model.module)
            self.model.update_from_file(network.config_file)

        # learning rate options
        if args_['learning_rate'] is not None:
            self.train.learning_rate.value = args_['learning_rate']

        if self.train.learning_rate.schedule:
            self.train.epoch.nrof_epochs = self.train.learning_rate.schedule[-1][0]

        if self.validate:
            self.validate.batch_size = self.batch_size
            self.validate.image.size = self.image.size
            self.validate.image.standardization = self.image.standardization

        if not self.validate.validate.file:
            self.validate.validate.file = Path(self.model.path).expanduser().joinpath('report.txt')

        # write arguments and store some git revision info in a text files in the log directory
        ioutils.write_arguments(self, self.logs.joinpath('arguments.yaml'))
        ioutils.store_revision_info(self.logs, sys.argv)


class DefaultConfig:
    def __init__(self):
        self.model = src_dir.joinpath('models', '20190822-033520')
        self.pretrained_checkpoint = src_dir.joinpath('models', '20190822-033520', 'model-20190822-033520.ckpt-275')

        # image size (height, width) in pixels
        self.image_size = 160

        # batch size (number of images to process in a batch
        self.batch_size = 100
