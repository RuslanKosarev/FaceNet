# coding: utf-8
__author__ = 'Ruslan N. Kosarev'

from tqdm import tqdm
from pathlib import Path
from loguru import logger

import tensorflow as tf
import numpy as np
import random


def tf_dataset_api(files, labels, loader, batch_size, buffer_size=None, repeat=False):

    if buffer_size is not None:
        data = list(zip(files, labels))
        np.random.shuffle(data)
        files, labels = map(list, zip(*data))

    images = tf.data.Dataset.from_tensor_slices(files).map(loader, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    labels = tf.data.Dataset.from_tensor_slices(labels)

    ds = tf.data.Dataset.zip((images, labels))

    if buffer_size is not None:
        ds = ds.shuffle(buffer_size=buffer_size*batch_size, reshuffle_each_iteration=True)

    if repeat:
        ds = ds.repeat()

    ds = ds.batch(batch_size=batch_size)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    info = (f'{ds}\n' +
            f'batch size: {batch_size}\n' +
            f'buffer size: {buffer_size}\n' +
            f'cardinality: {ds.cardinality()}')

    logger.info('\n' + info)

    return ds


class ImageClass:
    """
    Stores the paths to images for a given class
    """

    def __init__(self, config):

        if not config.path:
            raise ValueError('Path to download dataset does not specified.')

        self.path = Path(config.path).expanduser()
        self.name = self.path.stem

        if not self.path.exists():
            raise ValueError(f'Directory {self.path} does not exist')

        files = list(self.path.glob('*'))

        if config.max_nrof_images:
            if len(files) > config.max_nrof_images:
                files = np.random.choice(files, size=config.max_nrof_images, replace=False)

        self.files = [str(f) for f in files]
        self.files.sort()

    def __repr__(self):
        return f'{self.__class__.__name__} ({self.name}/{self.nrof_images})'

    @property
    def nrof_images(self):
        return len(self.files)

    @property
    def nrof_pairs(self):
        return self.nrof_images * (self.nrof_images - 1) // 2


class Database:
    def __init__(self, config):

        if not config.path:
            raise ValueError('Path to download dataset does not specified.')

        self.path = Path(config.path).expanduser()
        if not self.path.exists():
            raise ValueError(f'Directory {self.path} does not exist')
        print('Download data set from {}'.format(self.path))

        dirs = [p for p in self.path.glob('*') if p.is_dir()]
        if config.nrof_classes:
            if len(dirs) > config.nrof_classes:
                dirs = np.random.choice(dirs, size=config.nrof_classes, replace=False)
        dirs.sort()

        self.classes = []

        with tqdm(total=len(dirs)) as bar:
            for idx, path in enumerate(dirs):
                config.path = path
                images = ImageClass(config)

                if images.nrof_images > 0:
                    self.classes.append(images)

                bar.set_postfix_str(f'{str(images)}')
                bar.update()

        logger.info(self)

    def __repr__(self):
        """Representation of the database"""
        return (f'{self.__class__.__name__}\n' +
                f'{self.path}\n' +
                f'Number of classes {self.nrof_classes} \n' +
                f'Number of images {self.nrof_images}\n' +
                f'Minimal number of images in class {self.min_nrof_images}\n' +
                f'Maximal number of images in class {self.max_nrof_images}\n')

    @property
    def files(self):
        files = []
        for cls in self.classes:
            files += cls.files
        return files

    @property
    def labels(self):
        labels = []
        for idx, cls in enumerate(self.classes):
            labels += [idx] * cls.nrof_images
        return np.array(labels)

    @property
    def min_nrof_images(self):
        return min(cls.nrof_images for cls in self.classes)

    @property
    def max_nrof_images(self):
        return max(cls.nrof_images for cls in self.classes)

    @property
    def nrof_classes(self):
        return len(self.classes)

    @property
    def nrof_images(self):
        return sum(cls.nrof_images for cls in self.classes)

    @property
    def nrof_images_per_class(self):
        return [cls.nrof_images for cls in self.classes]

    def tf_dataset_api(self, loader, batch_size, buffer_size=None, repeat=False):
        return tf_dataset_api(self.files,
                              self.labels,
                              loader,
                              batch_size,
                              buffer_size=buffer_size,
                              repeat=repeat)
