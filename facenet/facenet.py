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


def center_loss(features, label, alfa, nrof_classes):
    """Center loss based on the paper "A Discriminative Feature Learning Approach for Deep Face Recognition"
       (http://ydwen.github.io/papers/WenECCV16.pdf)
    """
    nrof_features = features.get_shape()[1]
    centers = tf.get_variable('centers', [nrof_classes, nrof_features], dtype=tf.float32,
        initializer=tf.constant_initializer(0), trainable=False)
    label = tf.reshape(label, [-1])
    centers_batch = tf.gather(centers, label)
    diff = (1 - alfa) * (centers_batch - features)
    centers = tf.scatter_sub(centers, label, diff)
    with tf.control_dependencies([centers]):
        loss = tf.reduce_mean(tf.square(features - centers_batch))
    return loss, centers


def split_embeddings(embarray: np.array, labels: np.array) -> List[np.array]:
    embeddings = []

    for label in np.unique(labels):
        embeddings.append(embarray[label == labels])

    return embeddings


class Embeddings:
    def __init__(self, config):
        self.config = config
        self.file = Path(config.path).expanduser()

        embeddings = h5utils.read(self.file, 'embeddings')
        labels = h5utils.read(self.file, 'labels')

        self.embeddings = split_embeddings(embeddings, labels)

        if self.config.nrof_classes:
            if self.nrof_classes > self.config.nrof_classes:
                labels = [_ for _ in range(self.nrof_classes)]
                labels = random.sample(labels, self.config.nrof_classes)

                self.embeddings = [self.embeddings[label] for label in labels]

        if self.config.max_nrof_images:
            for idx, emb in enumerate(self.embeddings):
                nrof_images = emb.shape[0]

                if nrof_images > self.config.max_nrof_images:
                    labels = [_ for _ in range(nrof_images)]
                    labels = random.sample(labels, self.config.max_nrof_images)

                    self.embeddings[idx] = self.embeddings[idx][labels, :]

    def __repr__(self):
        """Representation of the embeddings"""
        data = [len(e) for e in self.embeddings]

        embeddings = np.concatenate(self.embeddings, axis=0)
        norm = np.linalg.norm(embeddings, axis=1)

        info = (f'{self.__class__.__name__}\n' +
                f'Input file {self.file}\n' +
                f'Number of classes {self.nrof_classes} \n' +
                f'Number of images {self.nrof_images}\n' +
                f'Minimal number of images in class {min(data)}\n' +
                f'Maximal number of images in class {max(data)}\n' +
                '\n' +
                f'Minimal embedding {np.min(norm)}\n' +
                f'Maximal embedding {np.max(norm)}\n' +
                f'Mean embedding {np.mean(norm)}\n'
                )

        return info

    @property
    def nrof_classes(self):
        return len(self.embeddings)

    @property
    def nrof_images(self):
        return sum([len(e) for e in self.embeddings])

    @property
    def length(self):
        return self.embeddings[0].shape[1]

    def data(self, normalize=False):

        embeddings = self.embeddings

        if normalize:
            for idx in range(self.nrof_classes):
                embeddings[idx] /= np.linalg.norm(self.embeddings[idx], axis=1, keepdims=True)

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
