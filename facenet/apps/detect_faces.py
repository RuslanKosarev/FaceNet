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

    if not options.h5file.exists():
        h5_keys = []
    else:
        with h5py.File(str(options.h5file), mode='r') as hf:
            h5_keys = list(hf.keys())

    for cls in tqdm(dbase.classes):
        key = re.sub('[^A-Za-z0-9]', '', cls.name)

        if key not in h5_keys:
            df = detection.detect_faces(cls.files, detector)
            if not df.empty:
                df.to_hdf(options.h5file, key=key, mode='a')
        else:
            df = pd.read_hdf(options.h5file, key=key)

        if df.empty:
            continue

        df = df[df['size'] >= options.image.min_face_size]

        for image_path in cls.files:
            image_path = Path(image_path)
            image_index = f'{image_path.parent.name}/{image_path.stem}'

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
