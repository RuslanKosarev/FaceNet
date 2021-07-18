# coding:utf-8

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import math

from facenet import ioutils


class BoundingBox:
    def __init__(self, left, top, width, height, confidence=None):
        self.left = int(np.round(left))
        self.right = int(np.round(left + width))

        self.top = int(np.round(top))
        self.bottom = int(np.round(top + height))

        self.width = self.right - self.left
        self.height = self.bottom - self.top
        self.confidence = confidence

    def __repr__(self):
        return f'left {self.left}, top {self.top}, width {self.width}, height {self.height}, confidence {self.confidence}'


def image_processing(image, box, options):
    if not isinstance(image, Image.Image):
        raise ValueError('Input must be PIL.Image')

    left = box['left']
    right = box['right']
    top = box['top']
    bottom = box['bottom']

    if options.margin:
        width = right - left
        height = bottom - top
        w_margin = round(width * options.margin / 2)
        h_margin = round(height * options.margin / 2)
    else:
        w_margin = 0
        h_margin = 0

    # crop image
    if options.crop:
        image = image.crop((left - w_margin, top - h_margin, right + w_margin, bottom + h_margin))

    # resize image
    if options.resize:
        width = math.ceil(options.size + options.size * options.margin)
        height = math.ceil(options.size + options.size * options.margin)
        size = (width, height)

        image = image.resize(size, Image.ANTIALIAS)

    return image


def detect_faces(files, detector):

    df = pd.DataFrame()

    for image_path in files:
        image_path = Path(image_path)
        key = f'{image_path.parent.name}/{image_path.stem}'

        try:
            # this function returns PIL.Image object
            img = ioutils.read_image(image_path)
            img = ioutils.pil2array(img)
        except Exception as e:
            continue

        boxes = detector.detect_faces(img)

        if len(boxes) != 1:
            continue

        box = boxes[0]['box']
        confidence = boxes[0]['confidence']
        box = BoundingBox(left=box[0], top=box[1], width=box[2], height=box[3], confidence=confidence)

        series = pd.Series({
            'left': box.left,
            'right': box.right,
            'top': box.top,
            'bottom': box.bottom,
            'shape0': img.shape[0],
            'shape1': img.shape[1],
            'shape2': img.shape[2] if img.ndim == 3 else None,
            'confidence': confidence,
        }, name=key)

        df = pd.concat([df, series], axis=1)

    df = df.transpose()

    return df
