"""Validate a face recognizer.
"""
# MIT License
#
# Copyright (c) 2020 SMedX

import click
from pathlib import Path
from loguru import logger

import tensorflow as tf

from facenet import config, facenet, tfutils, ioutils, h5utils, logging
from facenet.dataset import Database


@click.command()
@click.option('--config', default=None, type=Path,
              help='Path to yaml config file with used options for the application.')
def main(**cfg):
    cfg = config.evaluate_embeddings(cfg)
    logging.configure_logging(cfg.logs)

    loader = facenet.ImageLoader(config=cfg.image)
    dataset = Database(cfg.dataset)
    tf_dataset = dataset.tf_dataset_api(
        loader,
        batch_size=cfg.batch_size
    )

    model = tf.keras.models.load_model(cfg.model.path, custom_objects=None, compile=True, options=None)

    embeddings, labels = facenet.evaluate_embeddings(model, tf_dataset)

    h5utils.write(cfg.h5file, 'embeddings', embeddings)
    h5utils.write(cfg.h5file, 'labels', labels)

    logger.info('Output h5 file: %s', cfg.outfile)
    logger.info('Number of examples: %s', dataset.nrof_images)


if __name__ == '__main__':
    main()
