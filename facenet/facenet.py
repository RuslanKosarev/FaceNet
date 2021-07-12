"""Functions for building the face recognition network.
"""

from typing import List
from tqdm import tqdm
from loguru import logger
from pathlib import Path

import random
import numpy as np
import tensorflow as tf

from facenet import nodes, h5utils, FaceNet


def inputs(config):
    return tf.keras.Input([config.size, config.size, 3])


def softmax_cross_entropy_with_logits(logits, labels):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    return cross_entropy_mean


class ImageLoader:
    def __init__(self, config=None):
        self.height = config.size
        self.width = config.size

    def __call__(self, path):
        contents = tf.io.read_file(path)
        image = tf.image.decode_image(contents, channels=3)
        image = tf.image.resize_with_crop_or_pad(image, self.height, self.width)
        return image


class ImageProcessing(tf.keras.layers.Layer):
    def __init__(self, config):
        super().__init__()

        self.input_node_name = nodes['input']['name']

        self.config = config
        self.image_size = tf.constant([self.config.size, self.config.size], name='image_size')
        self.eps = 1e-3

    def call(self, image_batch, **kwargs):
        image_batch = tf.identity(image_batch, name=self.input_node_name)
        image_batch = tf.cast(image_batch, dtype=tf.float32, name='float_image')
        image_batch = tf.image.resize(image_batch, size=self.image_size, name='resized_image')

        if self.config.normalization == 0:
            min_value = tf.math.reduce_min(image_batch, axis=[-1, -2, -3], keepdims=True)
            max_value = tf.math.reduce_max(image_batch, axis=[-1, -2, -3], keepdims=True)
            dynamic_range = tf.math.maximum(max_value - min_value, self.eps)

            image_batch = (2 * image_batch - (min_value + max_value)) / dynamic_range

        elif self.config.normalization == 1:
            image_batch = tf.image.per_image_standardization(image_batch)
        else:
            raise ValueError('Invalid image normalization algorithm')

        image_batch = tf.identity(image_batch, name=self.__class__.__name__ + '_output')

        return image_batch


def equal_batches_input_pipeline(embeddings, config):
    """
    Building equal batches input pipeline, for example, used in binary cross-entropy pipeline.

    :param embeddings: 
    :param config: 
    :return: 
    """""
    # if not config.nrof_classes_per_batch:
    #     config.nrof_classes_per_batch = len(embeddings)
    #
    # if not config.nrof_examples_per_class:
    #     config.nrof_examples_per_class = round(0.1*sum([len(embs) for embs in embeddings]) / len(embeddings))
    #     config.nrof_examples_per_class = max(config.nrof_examples_per_class, 1)

    logger.info('Building equal batches input pipeline.')
    logger.info('Number of classes per batch: {nrof_classes}', nrof_classes=config.nrof_classes_per_batch)
    logger.info('Number of examples per class: {nrof_examples}', nrof_examples=config.nrof_examples_per_class)

    def generator():
        while True:
            embs = []
            for embeddings_per_class in random.sample(embeddings, config.nrof_classes_per_batch):
                embs += random.sample(embeddings_per_class.tolist(), config.nrof_examples_per_class)
            yield embs

    ds = tf.data.Dataset.from_generator(generator, output_types=tf.float32)
    ds = ds.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x))

    batch_size = config.nrof_classes_per_batch * config.nrof_examples_per_class
    ds = ds.batch(batch_size)

    info = (f'{ds}\n' +
            f'batch size: {batch_size}\n' +
            f'cardinality: {ds.cardinality()}')

    logger.info('\n' + info)

    return ds


def evaluate_embeddings(model, tf_dataset):
    """
    Evaluate embeddings for given data set
    :param model:
    :param tf_dataset:
    :return:
    """

    embeddings = []
    labels = []

    for image_batch, label_batch in tqdm(tf_dataset):
        embeddings.append(model(image_batch))
        labels.append(label_batch)

    return np.concatenate(embeddings), np.concatenate(labels)


def split_embeddings(embeddings: np.array, labels: np.array) -> List[np.array]:
    """
    Split input 2D array of embeddings into list of embeddings, each element of list corresponds unique class

    :param embeddings:
    :param labels:
    :return:
    """

    embeddings = [embeddings[label == labels] for label in np.unique(labels)]

    return embeddings


class LearningRateScheduler:
    def __init__(self, config):
        self.config = config

        if self.config.value:
            self.default_value = self.config.value
        else:
            self.default_value = None

    def __call__(self, epoch):
        if self.default_value is not None:
            return self.default_value

        learning_rate = self.config.schedule[-1][1]

        for (epoch_, learning_rate) in self.config.schedule:
            if epoch < epoch_:
                break

        return learning_rate
