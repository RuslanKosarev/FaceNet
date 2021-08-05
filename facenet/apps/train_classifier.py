# coding:utf-8
""" Training a face classifier.
"""

import click
from pathlib import Path
from loguru import logger

import tensorflow as tf

from facenet.models.inception_resnet_v1 import InceptionResnetV1 as FaceNet
from facenet import facenet, config, callbacks
from facenet.dataset import Database


@click.command()
@click.option('-p', '--path', type=Path, default=None,
              help='Path to yaml config file with used options for the application.')
def main(path: Path):

    options = config.train_classifier(path)

    # ------------------------------------------------------------------------------------------------------------------
    # define train and test datasets
    loader = facenet.ImageLoader(options.image)

    train_dataset = Database(options.train.dataset)
    tf_train_dataset = train_dataset.tf_dataset_api(
        loader=loader,
        batch_size=options.batch_size,
        repeat=True,
        buffer_size=10
    )

    validate_dataset = Database(options.validate.dataset)
    tf_validate_dataset = validate_dataset.tf_dataset_api(
        loader=loader,
        batch_size=options.batch_size,
        repeat=False,
        buffer_size=None
    )
    # ------------------------------------------------------------------------------------------------------------------
    # initialize network
    model = FaceNet(
        input_shape=facenet.inputs(options.image),
        image_processing=facenet.ImageProcessing(options.image)
    )
    model.summary()

    kernel_regularizer = tf.keras.regularizers.deserialize(model.config.regularizer.kernel.as_dict)

    network = tf.keras.Sequential([
        model,
        tf.keras.layers.Dense(
            train_dataset.nrof_classes,
            activation=None,
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
            kernel_regularizer=kernel_regularizer,
            use_bias=True,
            bias_initializer='zeros',
            bias_regularizer=None,
            name='logits'
        )
    ])

    network(facenet.inputs(options.image))

    if options.model.checkpoint:
        checkpoint = Path(options.model.checkpoint).expanduser() / options.model.checkpoint.stem
        logger.info('Restore checkpoint %s', checkpoint)
        network.load_weights(checkpoint)

    # ------------------------------------------------------------------------------------------------------------------
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=options.outdir / options.outdir.stem,
        save_weights_only=True,
        verbose=True
    )

    learning_rate_callback = tf.keras.callbacks.LearningRateScheduler(
        facenet.LearningRateScheduler(config=options.train.learning_rate),
        verbose=True
    )

    validate_callbacks = callbacks.ValidateCallback(
        model,
        tf_validate_dataset,
        every_n_epochs=options.validate.every_n_epochs,
        max_nrof_epochs=options.train.epoch.max_nrof_epochs,
        config=options.validate
    )

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=options.logdir)

    network.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(epsilon=0.1)
    )

    network.fit(
        tf_train_dataset,
        epochs=options.train.epoch.max_nrof_epochs,
        steps_per_epoch=options.train.epoch.size,
        callbacks=[
            checkpoint_callback,
            learning_rate_callback,
            validate_callbacks,
            tensorboard_callback,
        ]
    )

    network.save(options.outdir)

    logger.info('Model and logs have been saved to the directory {path}', path=options.outdir)


if __name__ == '__main__':
    main()

