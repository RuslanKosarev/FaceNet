"""Train facenet classifier.
"""
# MIT License
#
# Copyright (c) 2020 SMedX

import click
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from tensorflow import keras

import tensorflow as tf
import numpy as np

from facenet import config, facenet, facerecognizer, ioutils, h5utils


class ConfusionMatrix:
    def __init__(self, embeddings, classifier):
        nrof_classes = len(embeddings)
        nrof_positive_class_pairs = nrof_classes
        nrof_negative_class_pairs = nrof_classes * (nrof_classes - 1) / 2

        tp = tn = fp = fn = 0

        for i in range(nrof_classes):
            for k in range(i):
                outs = classifier.predict(embeddings[i], embeddings[k])
                mean = np.mean(outs)

                fp += mean
                tn += 1 - mean

            outs = classifier.predict(embeddings[i])
            mean = np.mean(outs)

            tp += mean
            fn += 1 - mean

        tp /= nrof_positive_class_pairs
        fn /= nrof_positive_class_pairs

        fp /= nrof_negative_class_pairs
        tn /= nrof_negative_class_pairs

        self.classifier = classifier
        self.accuracy = (tp + tn) / (tp + fp + tn + fn)
        self.precision = tp / (tp + fp)
        self.tp_rate = tp / (tp + fn)
        self.tn_rate = tn / (tn + fp)

    def __repr__(self):
        return (f'{type(self).__name__}\n' +
                f'{str(self.classifier)}\n' +
                f'accuracy  {self.accuracy}\n' +
                f'precision {self.precision}\n' +
                f'tp rate   {self.tp_rate}\n' +
                f'tn rate   {self.tn_rate}\n')


def binary_cross_entropy_loss(logits, cfg):
    # define upper-triangle indices
    batch_size = cfg.nrof_classes_per_batch * cfg.nrof_examples_per_class
    triu_indices = [(i, k) for i, k in zip(*np.triu_indices(batch_size, k=1))]

    # compute labels for embeddings
    labels = []
    for i, k in triu_indices:
        if (i // cfg.nrof_examples_per_class) == (k // cfg.nrof_examples_per_class):
            # label 1 means inner class distance
            labels.append(1)
        else:
            # label 0 means across class distance
            labels.append(0)

    pos_weight = len(labels) / sum(labels) - 1

    logits = tf.gather_nd(logits, triu_indices)
    labels = tf.constant(labels, dtype=logits.dtype)

    # initialize cross entropy loss
    cross_entropy = tf.nn.weighted_cross_entropy_with_logits(labels, logits, pos_weight)
    loss = tf.reduce_mean(cross_entropy)

    return loss


@click.command()
@click.option('-m', '--model', type=Path, required=True,
              help='Path to directory with saved model.')
def main(model: Path):

    options = config.train_recognizer(model)

    h5file = list(model.glob('*.h5'))[0]
    embeddings = h5utils.read(h5file, 'embeddings')
    labels = h5utils.read(h5file, 'labels')

    embeddings = facenet.split_embeddings(embeddings, labels)

    options.recognizer.nrof_classes_per_batch = 16
    options.recognizer.nrof_examples_per_class = 16
    tf_train_dataset = facenet.equal_batches_input_pipeline(embeddings, options.recognizer)

    # define classifier
    input_shape = tf.keras.Input([embeddings[0].shape[1]])
    model = facerecognizer.FaceToFaceRecognizer(input_shape)
    model.summary()

    epochs = 100
    nrof_step = 100
    # optimizer = keras.optimizers.SGD(learning_rate=0.0001)
    optimizer = tf.keras.optimizers.Adam(epsilon=0.01)

    for epoch in range(epochs):
        for step, x_batch_train, in enumerate(tf_train_dataset):
            if step > nrof_step:
                break

            # Run the forward pass of the layer
            with tf.GradientTape() as tape:
                logits = model.call(x_batch_train)
                loss_value = binary_cross_entropy_loss(logits, options.recognizer)

            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

        print(f'Training loss (for one batch) at {epoch}: {loss_value}')
        print(model.alpha.numpy(), model.threshold.numpy())

    conf_mat = ConfusionMatrix(embeddings, model)
    print(conf_mat)


if __name__ == '__main__':
    main()
