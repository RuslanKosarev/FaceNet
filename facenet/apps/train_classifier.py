# coding:utf-8
"""Training a face recognizer with TensorFlow using softmax cross entropy loss
"""
# MIT License
# 
# Copyright (c) 2020 sMedX
# 

import click
from pathlib import Path
from loguru import logger

import tensorflow as tf

from facenet.models.inception_resnet_v1 import InceptionResnetV1 as FaceNet
from facenet import facenet, config, dataset, logging, callbacks


@click.command()
@click.option('--config', default=None, type=Path,
              help='Path to yaml config file with used options of the application.')
def main(**options):
    cfg = config.train_softmax(options)
    logging.configure_logging(cfg.logs)

    # ------------------------------------------------------------------------------------------------------------------
    # define train and test datasets
    loader = facenet.ImageLoader(config=cfg.image)

    train_dbase = dataset.Database(cfg.dataset)
    train_dataset = train_dbase.tf_dataset_api(loader,
                                               batch_size=cfg.batch_size,
                                               repeat=True,
                                               buffer_size=10)

    test_dbase = dataset.Database(cfg.validate.dataset)
    test_dataset = test_dbase.tf_dataset_api(loader,
                                             batch_size=cfg.batch_size,
                                             repeat=False,
                                             buffer_size=None)

    # ------------------------------------------------------------------------------------------------------------------
    # define network to train
    model = FaceNet(input_shape=facenet.inputs(cfg.image),
                    image_processing=facenet.ImageProcessing(cfg.image))
    model.summary()

    kernel_regularizer = tf.keras.regularizers.deserialize(model.config.regularizer.kernel.as_dict)

    network = tf.keras.Sequential([
        model,
        tf.keras.layers.Dense(train_dbase.nrof_classes,
                              activation=None,
                              kernel_initializer=tf.keras.initializers.GlorotUniform(),
                              kernel_regularizer=kernel_regularizer,
                              bias_initializer='zeros',
                              bias_regularizer=None,
                              name='logits')
    ])

    network(facenet.inputs(cfg.image))

    if cfg.model.checkpoint:
        checkpoint = cfg.model.checkpoint / cfg.model.checkpoint.stem
        print(f'Restore checkpoint {checkpoint}')
        network.load_weights(checkpoint)

    # ------------------------------------------------------------------------------------------------------------------
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=cfg.model.path / cfg.model.path.stem,
        save_weights_only=True,
        verbose=True
    )

    learning_rate_callback = tf.keras.callbacks.LearningRateScheduler(
        facenet.LearningRateScheduler(config=cfg.train.learning_rate),
        verbose=True
    )

    validate_callbacks = callbacks.ValidateCallback(model, test_dataset,
                                                    every_n_epochs=cfg.validate.every_n_epochs,
                                                    max_nrof_epochs=cfg.train.epoch.max_nrof_epochs,
                                                    config=cfg.validate)

    network.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(epsilon=0.1)
    )

    network.fit(
        train_dataset,
        epochs=cfg.train.epoch.max_nrof_epochs,
        steps_per_epoch=cfg.train.epoch.size,
        callbacks=[
            checkpoint_callback,
            learning_rate_callback,
            validate_callbacks,
        ]
    )
    network.save(cfg.model.path)

    print(f'Model and logs have been saved to the directory: {cfg.model.path}')


if __name__ == '__main__':
    main()

