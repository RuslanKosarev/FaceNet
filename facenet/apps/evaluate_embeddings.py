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

    dataset = Database(cfg.dataset)

    ioutils.write_text_log(cfg.logfile, dataset)
    embeddings = facenet.EvaluationOfEmbeddings(dataset, cfg)

    h5utils.write(cfg.outfile, 'embeddings', embeddings.embeddings)
    h5utils.write(cfg.outfile, 'labels', embeddings.labels)

    logger.info('Output file:', cfg.outfile)
    logger.info('Number of examples:', dataset.nrof_images)


if __name__ == '__main__':
    main()
