"""Validate a face recognizer.
"""
# MIT License
#
# Copyright (c) 2020 SMedX

import click
from pathlib import Path
import tensorflow as tf

from facenet import config, facenet, tfutils, ioutils, h5utils
from facenet.dataset import Database


@click.command()
@click.option('--config', default=None, type=Path,
              help='Path to yaml config file with used options for the application.')
def main(**cfg):
    cfg = config.evaluate_embeddings(cfg)

    dbase = Database(cfg.dataset)
    ioutils.write_text_log(cfg.logfile, dbase)
    print(dbase)

    embeddings = facenet.EvaluationOfEmbeddings(dbase, cfg)
    ioutils.write_text_log(cfg.logfile, dbase)
    print(embeddings)

    h5utils.write(cfg.outfile, 'embeddings', embeddings.embeddings)
    h5utils.write(cfg.outfile, 'labels', embeddings.labels)

    print('output file:', cfg.outfile)
    print('number of examples:', dbase.nrof_images)


if __name__ == '__main__':
    main()
