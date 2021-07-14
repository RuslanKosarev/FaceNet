"""Validate a face recognizer.
"""


import click
from pathlib import Path
from loguru import logger
import tensorflow as tf

from facenet import config, facenet, h5utils
from facenet.dataset import Database


@click.command()
@click.option('-p', '--path', type=Path, default=None,
              help='Path to yaml config file with used options for the application.')
def main(path: Path):

    options = config.evaluate_embeddings(path)

    loader = facenet.ImageLoader(config=options.image)
    dataset = Database(options.dataset)
    tf_dataset = dataset.tf_dataset_api(
        loader=loader,
        batch_size=options.batch_size
    )

    model = tf.keras.models.load_model(options.model.path, custom_objects=None, compile=True, options=None)
    model = model.get_layer('inception_resnet_v1')

    embeddings, labels = facenet.evaluate_embeddings(model, tf_dataset)

    h5utils.write(options.h5file, 'embeddings', embeddings)
    h5utils.write(options.h5file, 'labels', labels)

    logger.info('Output h5 file: {file}', file=options.h5file)
    logger.info('Shape of embeddings: {shape}', shape=embeddings.shape)


if __name__ == '__main__':
    main()
