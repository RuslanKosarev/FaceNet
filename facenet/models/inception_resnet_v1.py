"""Contains the definition of the Inception Resnet V1 architecture.
As described in http://arxiv.org/abs/1602.07261.
  Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning
  Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import ReLU, Conv2D, MaxPool2D, AvgPool2D, Dense, Flatten, Dropout, BatchNormalization

from facenet.config import Config

default_config = {
    'reduction_a': {
        'filters': [[384], [192, 192, 256]]
    },
    'reduction_b': {
        'filters': [[256, 384], [256, 256], [256, 256, 256]]
    },
    'block35': {
        'repeat': 5,
        'scale': 0.17,
        'activation': 'relu'
    },
    'block17': {
        'repeat': 10,
        'scale': 0.10,
        'activation': 'relu'
    },
    'block8_1': {
        'repeat': 5,
        'scale': 0.2,
        'activation': 'relu'
        },
    'block8_2': {
        'scale': 1.0,
        'activation': None
        },
    # outputs
    'output': {
        'size': 512
    },
}

regularizer = {
    'kernel': {
        'class_name': 'L2',
        'config': {'l2': 0.0005}
    },
    'activity': {
        'class_name': 'L1',
        'config': {'l1': 0}
    }
}

batch_normalization = {
    # 'momentum': 0.995,
    # 'epsilon': 0.001,
    'fused': False,
    'trainable': True,
    'center': True,
    'scale': False
}

kernel_regularizer = tf.keras.regularizers.L2(0.0005)
kernel_initializer = tf.keras.initializers.GlorotUniform()


def check_input_config(cfg=None):
    if cfg is None:
        cfg = Config(default_config)

    if not cfg.batch_normalization:
        cfg.batch_normalization = Config(batch_normalization)

    if not cfg.regularizer:
        cfg.regularizer = Config(regularizer)

    return cfg


# Inception-Resnet-A
class Block35(keras.layers.Layer):
    """Builds the 35x35 resnet block."""
    def __init__(self, config):
        super().__init__()
        self.config = check_input_config(config)
        self.mixed_activation = tf.keras.activations.deserialize('relu')

        self.scale = tf.Variable(0, name='Block35/scale', trainable=True, dtype=tf.float32)

        self.tower_conv0 = tf.keras.Sequential([
            Conv2D(32, 1, strides=1, padding='same', activation=None, use_bias=False,
                   kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer,
                   name='Conv2d_1x1'),
            BatchNormalization(**self.config.batch_normalization.as_dict),
            ReLU()
        ])

        self.tower_conv1 = tf.keras.Sequential([
            Conv2D(32, 1, strides=1, padding='same', activation=None, use_bias=False,
                   kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer,
                   name='Conv2d_0a_1x1'),
            BatchNormalization(**self.config.batch_normalization.as_dict),
            ReLU(),
            Conv2D(32, 3, strides=1, padding='same', activation=None, use_bias=False,
                   kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer,
                   name='Conv2d_0b_3x3'),
            BatchNormalization(**self.config.batch_normalization.as_dict),
            ReLU()
        ])

        self.tower_conv2 = tf.keras.Sequential([
            Conv2D(32, 1, strides=1, padding='same', activation=None, use_bias=False,
                   kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer,
                   name='Conv2d_0a_1x1'),
            BatchNormalization(**self.config.batch_normalization.as_dict),
            ReLU(),
            Conv2D(32, 3, strides=1, padding='same', activation=None, use_bias=False,
                   kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer,
                   name='Conv2d_0b_3x3'),
            BatchNormalization(**self.config.batch_normalization.as_dict),
            ReLU(),
            Conv2D(32, 3, strides=1, padding='same', activation=None, use_bias=False,
                   kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer,
                   name='Conv2d_0c_3x3'),
            BatchNormalization(**self.config.batch_normalization.as_dict),
            ReLU()
        ])

        self.up = Conv2D(256, 1, strides=1, padding='same', activation=None, use_bias=True,
                         kernel_initializer=kernel_initializer,
                         kernel_regularizer=kernel_regularizer,
                         name='Conv2d_1x1')

    def call(self, net, **kwargs):
        mixed = tf.concat([self.tower_conv0(net),
                           self.tower_conv1(net),
                           self.tower_conv2(net)], 3)

        net += self.scale * self.up(mixed)

        if self.mixed_activation:
            net = self.mixed_activation(net)

        return net


class Block17(keras.layers.Layer):
    """Builds the 17x17 resnet block."""
    def __init__(self, config):
        super().__init__()
        self.config = check_input_config(config)
        self.mixed_activation = tf.keras.activations.deserialize('relu')

        self.scale = tf.Variable(0, name='Block17/scale', trainable=True, dtype=tf.float32)

        self.tower_conv0 = tf.keras.Sequential([
            Conv2D(128, 1, strides=1, padding='same', activation=None, use_bias=False,
                   kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer,
                   name='Conv2d_1x1'),
            BatchNormalization(**self.config.batch_normalization.as_dict),
            ReLU()
        ])

        self.tower_conv1 = tf.keras.Sequential([
            Conv2D(128, 1, strides=1, padding='same', activation=None, use_bias=False,
                   kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer,
                   name='Conv2d_0a_1x1'),
            BatchNormalization(**self.config.batch_normalization.as_dict),
            ReLU(),
            Conv2D(128, (1, 7), strides=1, padding='same', activation=None, use_bias=False,
                   kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer,
                   name='Conv2d_0b_1x7'),
            BatchNormalization(**self.config.batch_normalization.as_dict),
            ReLU(),
            Conv2D(128, (7, 1), strides=1, padding='same', activation=None, use_bias=False,
                   kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer,
                   name='Conv2d_0c_7x1'),
            BatchNormalization(**self.config.batch_normalization.as_dict),
            ReLU()
        ])

        self.up = Conv2D(896, 1, strides=1, padding='same', activation=None, use_bias=True,
                         kernel_initializer=kernel_initializer,
                         kernel_regularizer=kernel_regularizer,
                         name='Conv2d_1x1')

    def call(self, net, **kwargs):
        mixed = tf.concat([self.tower_conv0(net),
                           self.tower_conv1(net)], 3)

        net += self.scale * self.up(mixed)

        if self.mixed_activation:
            net = self.mixed_activation(net)

        return net


# Inception-Resnet-C
class Block8(keras.layers.Layer):
    """Builds the 8x8 resnet block."""
    def __init__(self, config):
        super().__init__()
        self.config = check_input_config(config)
        self.mixed_activation = tf.keras.activations.deserialize(self.config.activation)

        self.scale = tf.Variable(0, name='Block8/scale', trainable=True, dtype=tf.float32)

        self.tower_conv0 = tf.keras.Sequential([
            Conv2D(192, 1, strides=1, padding='same', activation=None, use_bias=False,
                   kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer,
                   name='Conv2d_1x1'),
            BatchNormalization(**self.config.batch_normalization.as_dict),
            ReLU()
        ])

        self.tower_conv1 = tf.keras.Sequential([
            Conv2D(192, 1, strides=1, padding='same', activation=None, use_bias=False,
                   kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer,
                   name='Conv2d_0a_1x1'),
            BatchNormalization(**self.config.batch_normalization.as_dict),
            ReLU(),
            Conv2D(192, (1, 3), strides=1, padding='same', activation=None, use_bias=False,
                   kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer,
                   name='Conv2d_0b_1x3'),
            BatchNormalization(**self.config.batch_normalization.as_dict),
            ReLU(),
            Conv2D(192, (3, 1), strides=1, padding='same', activation=None, use_bias=False,
                   kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer,
                   name='Conv2d_0c_3x1'),
            BatchNormalization(**self.config.batch_normalization.as_dict),
            ReLU()
        ])

        self.up = Conv2D(1792, 1, strides=1, padding='same', activation=None, use_bias=True,
                         kernel_initializer=kernel_initializer,
                         kernel_regularizer=kernel_regularizer,
                         name='Conv2d_1x1')

    def call(self, net, **kwargs):
        mixed = tf.concat([self.tower_conv0(net),
                           self.tower_conv1(net)], 3)

        net += self.scale * self.up(mixed)

        if self.mixed_activation:
            net = self.mixed_activation(net)

        return net


class ReductionA(tf.keras.layers.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = check_input_config(config)

        filters = self.config.filters[0]

        self.tower_conv0 = tf.keras.Sequential([
            Conv2D(filters[0], 3, strides=2, padding='valid', activation=None, use_bias=False,
                   kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer,
                   name='Conv2d_1a_3x3'),
            BatchNormalization(**self.config.batch_normalization.as_dict),
            ReLU()
        ])

        filters = self.config.filters[1]

        self.tower_conv1 = tf.keras.Sequential([
            Conv2D(filters[0], 1, strides=1, padding='same', activation=None, use_bias=False,
                   kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer,
                   name='Conv2d_0a_1x1'),
            BatchNormalization(**self.config.batch_normalization.as_dict),
            ReLU(),
            Conv2D(filters[1], 3, strides=1, padding='same', activation=None, use_bias=False,
                   kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer,
                   name='Conv2d_0b_3x3'),
            BatchNormalization(**self.config.batch_normalization.as_dict),
            ReLU(),
            Conv2D(filters[2], 3, strides=2, padding='valid', activation=None, use_bias=False,
                   kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer,
                   name='Conv2d_1a_3x3'),
            BatchNormalization(**self.config.batch_normalization.as_dict),
            ReLU()
        ])

        self.tower_pool = MaxPool2D(3, strides=2, padding='valid', name='MaxPool_1a_3x3')

    def call(self, net, **kwargs):
        net = tf.concat([self.tower_conv0(net),
                         self.tower_conv1(net),
                         self.tower_pool(net)], 3)
        return net


class ReductionB(tf.keras.layers.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = check_input_config(config)

        filters = self.config.filters[0]
        self.tower_conv0 = tf.keras.Sequential([
            Conv2D(filters[0], 1, strides=1, padding='same', activation=None, use_bias=False,
                   kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer,
                   name='Conv2d_0a_1x1'),
            BatchNormalization(**self.config.batch_normalization.as_dict),
            ReLU(),
            Conv2D(filters[1], 3, strides=2, padding='valid', activation=None, use_bias=False,
                   kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer,
                   name='Conv2d_1a_3x3'),
            BatchNormalization(**self.config.batch_normalization.as_dict),
            ReLU()
        ])

        filters = self.config.filters[1]
        self.tower_conv1 = tf.keras.Sequential([
            Conv2D(filters[0], 1, strides=1, padding='same', activation=None, use_bias=False,
                   kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer,
                   name='Conv2d_0a_1x1'),
            BatchNormalization(**self.config.batch_normalization.as_dict),
            ReLU(),
            Conv2D(filters[1], 3, strides=2, padding='valid', activation=None, use_bias=False,
                   kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer,
                   name='Conv2d_1a_3x3'),
            BatchNormalization(**self.config.batch_normalization.as_dict),
            ReLU()
        ])

        filters = self.config.filters[2]
        self.tower_conv2 = tf.keras.Sequential([
            Conv2D(filters[0], 1,  strides=1, padding='same', activation=None, use_bias=False,
                   kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer,
                   name='Conv2d_0a_1x1'),
            BatchNormalization(**self.config.batch_normalization.as_dict),
            ReLU(),
            Conv2D(filters[1], 3,  strides=1, padding='same', activation=None, use_bias=False,
                   kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer,
                   name='Conv2d_0b_3x3'),
            BatchNormalization(**self.config.batch_normalization.as_dict),
            ReLU(),
            Conv2D(filters[2], 3, strides=2, padding='valid', activation=None, use_bias=False,
                   kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer,
                   name='Conv2d_1a_3x3'),
            BatchNormalization(**self.config.batch_normalization.as_dict),
            ReLU()
        ])

        self.tower_pool = MaxPool2D(3, strides=2, padding='valid', name='MaxPool_1a_3x3')

    def call(self, net, **kwargs):
        net = tf.concat([self.tower_conv0(net),
                         self.tower_conv1(net),
                         self.tower_conv2(net),
                         self.tower_pool(net)], 3)

        return net


class InceptionResnetV1(keras.Model):
    def __init__(self, input_shape, image_processing, config=None):
        super().__init__()
        self.config = check_input_config(config)

        self.image_processing = image_processing

        self.conv2d = tf.keras.Sequential(name='conv2d', layers=[
            Conv2D(32, 3, strides=2, padding='valid', use_bias=False, activation=None,
                   kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer,
                   name='Conv2d_1a_3x3'
                   ),
            BatchNormalization(**self.config.batch_normalization.as_dict),
            ReLU(),
            Conv2D(32, 3, strides=1, padding='valid', use_bias=False, activation=None,
                   kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer,
                   name='Conv2d_2a_3x3'
                   ),
            BatchNormalization(**self.config.batch_normalization.as_dict),
            ReLU(),
            Conv2D(64, 3, strides=1, padding='valid', use_bias=False, activation=None,
                   kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer,
                   name='Conv2d_2b_3x3'
                   ),
            BatchNormalization(**self.config.batch_normalization.as_dict),
            ReLU(),
            MaxPool2D(3, strides=2, padding='valid', name='MaxPool_3a_3x3'),
            Conv2D(80, 1, strides=1, padding='valid', use_bias=False, activation=None,
                   kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer,
                   name='Conv2d_3b_1x1',
                   ),
            BatchNormalization(**self.config.batch_normalization.as_dict),
            ReLU(),
            Conv2D(192, 3, strides=1, padding='valid', use_bias=False, activation=None,
                   kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer,
                   name='Conv2d_4a_3x3',
                   ),
            BatchNormalization(**self.config.batch_normalization.as_dict),
            ReLU(),
            Conv2D(256, 3, strides=2, padding='valid', use_bias=False, activation=None,
                   kernel_initializer=kernel_initializer,
                   kernel_regularizer=kernel_regularizer,
                   name='Conv2d_4b_3x3'),
            BatchNormalization(**self.config.batch_normalization.as_dict),
            ReLU()
        ])

        # repeat block35
        config = self.config.block35
        layers = [Block35(config=config) for _ in range(config.repeat)]
        self.repeat_block35 = tf.keras.Sequential(layers=layers, name='block35')

        # reduction a
        self.reduction_a = ReductionA(self.config.reduction_a)

        # repeat block17
        config = self.config.block17
        layers = [Block17(config=config) for _ in range(config.repeat)]
        self.repeat_block17 = tf.keras.Sequential(layers=layers, name='block17')

        # reduction b
        self.reduction_b = ReductionB(self.config.reduction_b)

        # repeat block8
        config = self.config.block8_1
        layers = [Block8(config=config) for _ in range(config.repeat)]
        self.repeat_block8 = tf.keras.Sequential(layers=layers, name='block8')

        self.block8 = Block8(config=self.config.block8_2)

        # self.features = Features(config['features'])
        config = self.config.output
        activity_regularizer = tf.keras.regularizers.deserialize(self.config.regularizer.activity.as_dict)

        self.features = tf.keras.Sequential(name='features', layers=[
            AvgPool2D([3, 3], padding='valid', name='AvgPool_1a_8x8'),
            Flatten(),
            Dense(config.size, activation=None, use_bias=False,
                  kernel_initializer=kernel_initializer,
                  kernel_regularizer=kernel_regularizer,
                  activity_regularizer=activity_regularizer,
                  name='logits'),
            BatchNormalization(**self.config.batch_normalization.as_dict)
        ])

        self.custom_layers = (
            self.image_processing,
            self.conv2d,
            self.repeat_block35,
            self.reduction_a,
            self.repeat_block17,
            self.reduction_b,
            self.repeat_block8,
            self.block8,
            self.features
        )

        self(input_shape)

    def call(self, inputs, training=False, **kwargs):
        # evaluate output of model
        output = inputs
        for layer in self.custom_layers:
            output = layer(output)

        # normalize embeddings
        if not training:
            output = tf.nn.l2_normalize(output, axis=1, epsilon=1e-10, name='embedding')

        return output

    def summary(self, line_length=None, positions=None, print_fn=None):
        super().summary(line_length, positions, print_fn)

        print('Total variables: ', len(self.variables))
        print('Trainable variables: ', len(self.trainable_variables))
        print('Non-trainable variables: ', len(self.non_trainable_variables))

