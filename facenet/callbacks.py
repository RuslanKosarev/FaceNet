# coding: utf-8
__author__ = 'Ruslan N. Kosarev'

from loguru import logger

import numpy as np
import tensorflow as tf

from facenet import facenet
from facenet import statistics


class ValidateCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, dataset, every_n_epochs, max_nrof_epochs, config):
        super().__init__()
        self._model = model
        self.dataset = dataset
        self.config = config
        self.every_n_epochs = every_n_epochs
        self.max_nrof_epochs = max_nrof_epochs

    def on_epoch_end(self, epoch, logs=None):
        epoch1 = epoch + 1

        if epoch1 % self.every_n_epochs == 0 or epoch1 == self.max_nrof_epochs:
            logger.info(f'perform validation for epoch {epoch1}')

            embeddings, labels = facenet.evaluate_embeddings(self._model, self.dataset)
            statistics.FaceToFaceValidation(embeddings, labels, self.config)
