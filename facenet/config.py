# coding: utf-8
__author__ = 'Ruslan N. Kosarev'

import sys
from typing import Union, List, Tuple
from pathlib import Path
from datetime import datetime
from omegaconf import OmegaConf, DictConfig

import random
import numpy as np
import tensorflow as tf

from facenet import ioutils, logging

PathType = Union[Path, str]

# directory for default configs
default_config_dir = Path(__file__).parents[0].joinpath('apps', 'configs')
default_config = default_config_dir / 'config.yaml'

# directory for user's configs
user_config_dir = Path(__file__).parents[1] / 'configs'

# directory for default trained model
default_model_path = Path(__file__).parents[1] / 'models/default'

user_config = user_config_dir / 'config.yaml'
default_train_config = default_config_dir / 'train_config.yaml'
default_evaluate_embeddings_config = default_config_dir / 'evaluate_embeddings.yaml'
default_train_recognizer_config = default_config_dir / 'train_recognizer.yaml'


def subdir() -> str:
    return datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')


def application_name():
    return Path(sys.argv[0]).stem


def config_paths(app_file_name, custom_config_file=None):
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
    """
    set seed for random number generators

    :param seed:
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
#    tf.random.set_seed(seed)


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


def load_config(app_file_name, file=None, keys: Tuple[str, ...] = None):
    """Load configuration from the set of config files
    :param keys:
    :param app_file_name
    :param file: Optional path to the custom config file
    :return: The validated config in Config model instance
    """

    paths = config_paths(app_file_name, file)

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

    options = OmegaConf.to_container(cfg)
    options = Config(options)

    if keys is not None:
        options = Config({key: getattr(options, key) for key in keys})

    return options


def train_classifier(path: PathType):
    options = load_config(application_name(), path)

    set_seed(options.seed)

    options.outdir = Path(options.model.path).expanduser() / subdir()
    options.logdir = options.outdir / 'log'

    if options.model.checkpoint:
        options.model.checkpoint = Path(options.model.checkpoint).expanduser()

    if not options.train.epoch.max_nrof_epochs:
        options.train.epoch.max_nrof_epochs = options.train.learning_rate.schedule[-1][0]

    if options.validate:
        options.validate.batch_size = options.batch_size
        options.validate.image.size = options.image.size
        options.validate.image.standardization = options.image.standardization

    # write arguments and some git revision info
    ioutils.write_arguments(options.logdir, options)
    ioutils.write_revision_info(options.logdir)

    logging.configure_logging(options.logdir)

    return options


def train_recognizer(path: PathType):
    options = load_config(application_name(), path)

    set_seed(options.seed)

    if not options.train.epoch.max_nrof_epochs:
        options.train.epoch.max_nrof_epochs = options.train.learning_rate.schedule[-1][0]

    if not options.outdir:
        options.outdir = Path(options.model.path).expanduser() / 'recognition' / subdir()

    # write arguments and some git revision info
    ioutils.write_arguments(options.outdir, options)
    ioutils.write_revision_info(options.outdir)

    logging.configure_logging(options.outdir)

    return options


def evaluate_embeddings(options):
    keys = ('seed', 'batch_size', 'model', 'image', 'dataset')
    options = load_config(application_name(), options, keys)

    set_seed(options.seed)
    options.model.path = Path(options.model.path).expanduser()

    if not options.outdir:
        outdir = options.model.path
        options.outdir = Path(outdir).expanduser() / Path(options.dataset.path).stem
    options.outdir = Path(options.outdir).expanduser()

    if not options.h5file:
        options.h5file = 'embeddings.h5'
    options.h5file = options.outdir / options.h5file

    # write arguments and some git revision info
    ioutils.write_arguments(options.outdir, options)
    ioutils.write_revision_info(options.outdir)

    logging.configure_logging(options.outdir)

    return options


def detect_faces(options):
    options = load_config(application_name(), options)

    set_seed(options.seed)

    if not options.outdir:
        path = Path(options.dataset.path).expanduser()
        options.outdir = f'{path}_extracted_faces'
    options.outdir = Path(options.outdir).expanduser()

    options.h5file = options.outdir.joinpath('boxes.h5')
    options.h5file = options.outdir / options.h5file

    # write arguments and some git revision info
    ioutils.write_arguments(options.outdir, options)
    ioutils.write_revision_info(options.outdir)

    logging.configure_logging(options.outdir)

    return options








