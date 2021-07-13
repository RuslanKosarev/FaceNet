# coding:utf-8
__author__ = 'Ruslan N. Kosarev'

import numpy as np
import tensorflow as tf
from tensorflow import keras


class FaceToFaceDistanceClassifier:
    def __init__(self):
        """
        normalized distance between embeddings
        l = (norm(x) + norm(y))/2
        x1 = x / norm(x), y1 = y / norm(y)

        distance = (x1-y1,x1-y1) + pow((norm(x)-norm(y))/l, 2)
        """
        self.variables = {
            'alpha': tf.Variable(10, dtype=tf.float32, name='alpha'),
            'threshold': tf.Variable(1, dtype=tf.float32, name='threshold'),
            'theta': tf.Variable(1, dtype=tf.float32, name='theta')
        }

    def __call__(self, x, y=None):
        alpha = self.variable('alpha')
        threshold = self.variable('threshold')
        logits = tf.multiply(alpha, tf.subtract(threshold, self.distance(x, y)))
        return logits

    def __repr__(self):
        variables = {}
        for name in self.variables.keys():
            variables[name] = self.variable(name, mode='numpy')

        return (f'{self.__class__.__name__}\n'
                f'variables {variables}\n')

    def variable(self, name, mode=None):
        var = self.variables[name]
        if mode == 'numpy':
            var = tf.get_default_session().run(var)
        return var

    def distance(self, x, y):
        if y is None:
            y = x

        if isinstance(x, np.ndarray):
            theta = self.variable('theta', mode='numpy')

            y = np.transpose(y)

            norm_x = np.linalg.norm(x, axis=1, keepdims=True)
            norm_y = np.linalg.norm(y, axis=0, keepdims=True)
        else:
            theta = self.variable('theta')

            y = tf.transpose(y)

            norm_x = tf.linalg.norm(x, axis=1, keepdims=True)
            norm_y = tf.linalg.norm(y, axis=0, keepdims=True)

        x1 = x / norm_x
        y1 = y / norm_y

        # length = (norm_x + norm_y) / 2
        # dl = 1 - (norm_x / length) * (norm_y / length)
        # first order of theta - (x - x1, x - x1) + (y - y1, y - y1)
        # second order of theta - (y1 - x1, x1 - x) + (y1 - x1, y - y1) + (x1 - x, y - y1)
        # dist = 2 * (1 - x1 @ y1) + 2 * theta * dl * x1 @ y1 + 2 * theta * theta * dl

        dist = 2 * (1 - x1 @ y1) + theta*pow(2*(norm_x - norm_y)/(norm_x + norm_y), 2)

        return dist

    def predict(self, x, y=None):
        return self.distance(x, y) < self.variable('threshold', mode='numpy')


class FaceToFaceRecognizer(keras.Model):
    def __init__(self, input_shape):
        super().__init__()
        prefix = type(self).__name__

        self.alpha = tf.Variable(1, name=f'{prefix}/alpha', trainable=True, dtype=tf.float32)
        self.threshold = tf.Variable(1, name=f'{prefix}/threshold', trainable=True, dtype=tf.float32)

        self(input_shape) # noqa

    def call(self, x, y=None):
        logits = tf.multiply(self.alpha, tf.subtract(self.threshold, self.distance(x, y)))
        return logits

    @staticmethod
    def distance(x, y=None):
        x = tf.nn.l2_normalize(x, axis=1)

        if y is not None:
            y = tf.nn.l2_normalize(y, axis=1)
        else:
            y = x

        dist = 2 * (1 - x @ tf.transpose(y))

        return dist

    def predict(self, x, y=None):
        return tf.less(self.distance(x, y), self.threshold)
