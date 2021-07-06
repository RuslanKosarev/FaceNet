# coding:utf-8
__author__ = 'Ruslan N. Kosarev'

import numpy as np
from PIL import Image
import math


def image_processing(image, box, options):
    if not isinstance(image, Image.Image):
        raise ValueError('Input must be PIL.Image')

    w_margin = round(box.width * options.margin / 2)
    h_margin = round(box.height * options.margin / 2)

    cropped = image.crop((box.left - w_margin, box.top - h_margin, box.right + w_margin, box.bottom + h_margin))

    # compute size of the output image
    width = math.ceil(options.size + options.size * options.margin)
    height = math.ceil(options.size + options.size * options.margin)
    size = (width, height)

    # resize input image
    resized = cropped.resize(size, Image.ANTIALIAS)

    return resized


class BoundingBox:
    def __init__(self, left, top, width, height, confidence=None):
        self.left = int(np.round(left))
        self.right = int(np.round(left + width)) + 1

        self.top = int(np.round(top))
        self.bottom = int(np.round(top + height)) + 1

        self.width = self.right - self.left - 1
        self.height = self.bottom - self.top - 1
        self.confidence = confidence

    def info(self, mode=False):
        if mode is False:
            return '{}'.format([self.left, self.top, self.width, self.height, self.confidence])
        info = "left = {}, top = {}, width = {}, height = {}, confidence = {}".format(self.left, self.top, self.width, self.height, self.confidence)
        return info

    def __repr__(self):
        return self.info(mode=True)

    @property
    def left_upper(self):
        return self.left, self.top

    @property
    def right_lower(self):
        return self.right, self.bottom

    @property
    def confidence_as_string(self):
        return str(np.round(self.confidence, 3))


class MTCNN:
    def __init__(self):
        from mtcnn.mtcnn import MTCNN
        self.__detector = MTCNN().detect_faces
        self.mode = 'RGB'

    def detector(self, image):
        faces = self.__detector(image)
        bboxes = []

        for face in faces:
            box = face['box']
            bbox = BoundingBox(left=box[0], top=box[1], width=box[2], height=box[3], confidence=face['confidence'])
            bboxes.append(bbox)

        return bboxes


class FaceDetector:
    def __init__(self):

        obj = MTCNN()
        self.mode = obj.mode
        self.__detector = obj.detector

    def detect(self, image):
        return self.__detector(image)

    def __repr__(self):
        info = (f'class {self.__class__.__name__}\n' +
                f'detector type: {self.detector}')
        return info

