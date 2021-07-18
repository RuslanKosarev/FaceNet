""" Detect faces and store face thumbnails in the output directory."""

import click
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from mtcnn import MTCNN
from mtcnn_cv2 import MTCNN

from facenet import dataset
from facenet import config
from loguru import logger
import h5py

from facenet import detection


@click.command()
@click.option('-p', '--path', type=Path, default=None,
              help='Path to yaml config file with used options for the application.')
def main(path: Path):

    options = config.detect_faces(path)
    dbase = dataset.Database(options.dataset)

    detector = MTCNN(min_face_size=options.image.min_face_size)

    if not options.h5file.exists():
        h5_keys = []
    else:
        with h5py.File(str(options.h5file), mode='r') as hf:
            h5_keys = list(hf.keys())

    for cls in tqdm(dbase.classes):
        if cls.name not in h5_keys:
            df = detection.detect_faces(cls.files, detector)
            df.to_hdf(options.h5file, key=cls.name, mode='a')

    logger.info('output directory {outdir}', outdir=options.outdir)


if __name__ == '__main__':
    main()
