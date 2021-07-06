# coding: utf-8
__author__ = 'Ruslan N. Kosarev'

from pathlib import Path
from datetime import datetime

from omegaconf import OmegaConf

import random
import numpy as np
import tensorflow as tf

from facenet import ioutils

# directory for default configs
default_config_dir = Path(__file__).parents[0].joinpath('apps', 'configs')
default_config = default_config_dir / 'config.yaml'

# directory for user's configs
user_config_dir = Path(__file__).parents[1] / 'configs'

# directory for default trained model
default_model_path = Path(__file__).parents[1] / 'models/default'

user_config = user_config_dir / 'config.yaml'
default_train_classifier_config = default_config_dir / 'train_classifier.yaml'
default_train_recognizer_config = default_config_dir / 'train_recognizer.yaml'


def subdir() -> str:
    return datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')


def config_paths(app_file_name, custom_config_file):
    config_name = Path(app_file_name).stem + '.yaml'

    paths = [
        default_config,
        default_config_dir.joinpath(config_name),
        user_config,
        user_config_dir.joinpath(config_name)
    ]

    if custom_config_file is not None:
        paths.append(custom_config_file)

    return tuple(paths)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


class Config:
    """Object representing YAML settings as a dict-like object with values as fields
    """

    def __init__(self, dct=None):
        """Update config from dict
        :param dct: input object
        """
        if dct is None:
            dct = dict()

        for key, item in dct.items():
            if isinstance(item, dict):
                setattr(self, key, Config(item))
            else:
                setattr(self, key, item)

    def __repr__(self):
        shift = 3 * ' '

        def get_str(obj, ident=''):
            s = ''
            for key, item in obj.items():
                if isinstance(item, Config):
                    s += f'{ident}{key}: \n{get_str(item, ident=ident + shift)}'
                else:
                    s += f'{ident}{key}: {str(item)}\n'
            return s

        return get_str(self)

    def __getattr__(self, name):
        return self.__dict__.get(name, Config())

    def __bool__(self):
        return bool(self.__dict__)

    @property
    def as_dict(self):
        def as_dict(obj):
            s = {}
            for key, item in obj.items():
                if isinstance(item, Config):
                    item = as_dict(item)
                s[key] = item
            return s

        return as_dict(self)

    def items(self):
        return self.__dict__.items()

    def exists(self, name):
        return True if name in self.__dict__.keys() else False


class LoadConfigError(Exception):
    pass


def load_config(app_file_name, options):
    """Load configuration from the set of config files
    :param app_file_name
    :param options: Optional path to the custom config file
    :return: The validated config in Config model instance
    """

    paths = config_paths(app_file_name, options['config'])

    cfg = OmegaConf.create()
    new_cfg = None

    for config_path in paths:
        if not config_path.is_file():
            continue

        try:
            new_cfg = OmegaConf.load(config_path)
            cfg = OmegaConf.merge(cfg, new_cfg)
        except Exception as err:
            raise LoadConfigError(f"Cannot load configuration from '{config_path}'\n{err}")

    if new_cfg is None:
        raise LoadConfigError("The configuration has not been loaded.")

    cfg = OmegaConf.to_container(cfg)
    cfg = Config(cfg)

    return cfg


def train_classifier(options):
    cfg = load_config(default_train_classifier_config, options)

    path = Path(cfg.model.path).expanduser()
    cfg.model.path = path / subdir()

    cfg.logs = Config()
    cfg.logs.dir = cfg.model.path / 'logs'
    cfg.logs.file = cfg.model.path.stem + '.log'

    if cfg.model.checkpoint:
        cfg.model.checkpoint = Path(cfg.model.checkpoint).expanduser()

    if not cfg.train.epoch.max_nrof_epochs:
        cfg.train.epoch.max_nrof_epochs = cfg.train.learning_rate.schedule[-1][0]

    if cfg.validate:
        cfg.validate.batch_size = cfg.batch_size
        cfg.validate.image.size = cfg.image.size
        cfg.validate.image.standardization = cfg.image.standardization

    # set seed for random number generators
    set_seed(cfg.seed)

    # write arguments and store some git revision info in a text files in the log directory
    ioutils.write_arguments(cfg, cfg.logs.dir)
    ioutils.store_revision_info(cfg.logs.dir)

    return cfg


def train_recognizer(options):
    cfg = load_config(default_train_recognizer_config, options)

    cfg.classifier.path = Path(cfg.classifier.path).expanduser() / subdir()

    cfg.logdir = cfg.classifier.path
    cfg.logfile = cfg.logdir / 'log.txt'

    # set seed for random number generators
    set_seed(cfg.seed)

    # write arguments and store some git revision info in a text files in the log directory
    ioutils.write_arguments(cfg, cfg.logdir.joinpath(Path(app_file_name).stem + '.yaml'))
    ioutils.store_revision_info(cfg.logdir)

    return cfg


def evaluate_embeddings(app_file_name, options):
    cfg = load_config(app_file_name, options)

    if not cfg.model.path:
        cfg.model.path = default_model_path

    if cfg.suffix not in ('.h5', '.tfrecord'):
        raise ValueError('Invalid suffix for output file, must either be h5 or tfrecord.')

    cfg.outdir = Path(cfg.dataset.path + '_' + Path(cfg.model.path).stem)
    cfg.outdir = Path(cfg.outdir).expanduser()

    cfg.logdir = cfg.outdir
    cfg.logfile = cfg.outdir.joinpath('log.txt')
    cfg.outfile = cfg.outdir.joinpath('embeddings').with_suffix(cfg.suffix)

    # set seed for random number generators
    set_seed(cfg.seed)

    # write arguments and store some git revision info in a text files in the log directory
    ioutils.write_arguments(cfg, cfg.logdir.joinpath(Path(app_file_name).stem + '.yaml'))
    ioutils.store_revision_info(cfg.logdir)

    return cfg




