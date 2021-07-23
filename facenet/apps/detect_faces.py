""" Detect faces and store face thumbnails in the output directory."""

import click
from pathlib import Path
import re

import pandas as pd
from tqdm import tqdm
from mtcnn_cv2 import MTCNN

from facenet import dataset, ioutils
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

    detector = MTCNN()

    for cls in tqdm(dbase.classes):
        key = re.sub('[^A-Za-z0-9]', '', cls.name)

        df = detection.detect_faces(cls.files, detector)
        if df.empty:
            continue

        df.to_hdf(options.h5file, key=key, mode='a')
        df = df[df['size'] >= options.image.min_face_size]

        for image_path in cls.files:
            image_index = detection.df_index(image_path)

            if image_index in df.index:
                image = ioutils.read_image(image_path)
                image = detection.image_processing(image, df.loc[image_index], options.image)

                path = options.outdir / f'{image_index}.png'
                ioutils.write_image(image, path)

    options.dataset.path = options.outdir
    dataset.Database(options.dataset)
    logger.info('output directory {outdir}', outdir=options.outdir)


if __name__ == '__main__':
    main()
