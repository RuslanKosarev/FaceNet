# coding:utf-8

import re
from pathlib import Path
import pandas as pd
from PIL import Image
import math

from facenet import ioutils


class BoundingBox:
    def __init__(self, left, top, width, height, confidence=None):
        self.left = left
        self.right = left + width
        self.top = top
        self.bottom = top + height

        self.width = self.right - self.left
        self.height = self.bottom - self.top
        self.size = self.width * self.height

        self.confidence = confidence

    def __repr__(self):
        return f'left {self.left}, top {self.top}, width {self.width}, height {self.height}, confidence {self.confidence}'


def df_index(image_path):
    image_path = Path(image_path)

    parent = re.sub('[^A-Za-z0-9]', '', image_path.parent.name)
    name = re.sub('[^A-Za-z0-9]', '', image_path.stem)
    index = f'{parent}/{name}'

    return index


def image_processing(image, box, options):
    if not isinstance(image, Image.Image):
        raise ValueError('Input must be PIL.Image')

    left = box['left']
    right = box['right']
    top = box['top']
    bottom = box['bottom']

    if options.margin:
        margin = options.margin
    else:
        margin = 0

    width = right - left
    height = bottom - top
    w_margin = width * margin / 2
    h_margin = height * margin / 2

    # crop image
    if options.crop:
        image = image.crop((left - w_margin, top - h_margin, right + w_margin, bottom + h_margin))

    # resize image
    if options.resize:
        width = math.ceil(options.size + options.size * margin)
        height = math.ceil(options.size + options.size * margin)
        size = (width, height)

        image = image.resize(size, Image.ANTIALIAS)

    return image


def detect_faces(files, detector):

    df = pd.DataFrame()

    for image_path in files:
        image_path = Path(image_path)

        try:
            # this function returns PIL.Image object
            img = ioutils.read_image(image_path)
            img = ioutils.pil2array(img)
        except Exception as e:
            continue

        boxes = detector.detect_faces(img)

        if len(boxes) != 1:
            continue

        left, top, width, height = boxes[0]['box']
        confidence = boxes[0]['confidence']

        box = BoundingBox(left=left, top=top, width=width, height=height, confidence=confidence)

        series = pd.Series({
            'left': box.left,
            'right': box.right,
            'top': box.top,
            'bottom': box.bottom,
            'size': box.size,
            'shape0': img.shape[0],
            'shape1': img.shape[1],
            'shape2': img.shape[2] if img.ndim == 3 else None,
            'confidence': confidence,
        }, name=df_index(image_path))

        df = pd.concat([df, series], axis=1)

    df = df.transpose()

    return df
