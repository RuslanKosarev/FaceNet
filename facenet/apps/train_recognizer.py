"""Train facenet recognizer.
"""

import click
from pathlib import Path
from loguru import logger

import tensorflow as tf
import numpy as np

from facenet import config, facenet, facerecognizer, h5utils


class ConfusionMatrix:
    """
    Evaluate confusion matrix block by block for given model and embeddings
    """
    def __init__(self, embeddings, model):
        nrof_classes = len(embeddings)
        nrof_positive_class_pairs = nrof_classes
        nrof_negative_class_pairs = nrof_classes * (nrof_classes - 1) / 2

        tp = tn = fp = fn = 0

        # evaluate confusion matrix block by block
        for i in range(nrof_classes):
            outs = model.predict(embeddings[i])
            mean = tf.math.count_nonzero(outs).numpy() / (outs.shape[0] * outs.shape[1])

            tp += mean
            fn += 1 - mean

            for k in range(i):
                outs = model.predict(embeddings[i], embeddings[k])
                mean = tf.math.count_nonzero(outs).numpy() / (outs.shape[0] * outs.shape[1])

                fp += mean
                tn += 1 - mean

        tp /= nrof_positive_class_pairs
        fn /= nrof_positive_class_pairs

        fp /= nrof_negative_class_pairs
        tn /= nrof_negative_class_pairs

        self.accuracy = (tp + tn) / (tp + fp + tn + fn)
        self.precision = tp / (tp + fp)
        self.tp_rate = tp / (tp + fn)
        self.tn_rate = tn / (tn + fp)

    def __repr__(self):
        return (f'{type(self).__name__}\n' +
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
@click.option('--path', type=Path, default=None,
              help='Path to yaml config file with used options for the application.')
def main(path: Path):

    options = config.train_recognizer(path)

    embeddings = h5utils.read(options.embeddings.path, 'embeddings')
    labels = h5utils.read(options.embeddings.path, 'labels')

    embeddings = facenet.split_embeddings(embeddings, labels)

    tf_train_dataset = facenet.equal_batches_input_pipeline(embeddings, options.train)

    # initialize classifier
    input_shape = tf.keras.Input([embeddings[0].shape[1]])
    model = facerecognizer.FaceToFaceRecognizer(input_shape)
    model.summary()

    optimizer = tf.keras.optimizers.Adam(epsilon=0.1)
    schedule = options.train.learning_rate

    learning_rate_scheduler = facenet.LearningRateScheduler(config=schedule)

    for epoch in range(options.train.epoch.max_nrof_epochs):
        optimizer.lr.assign(learning_rate_scheduler(epoch))

        for step, x_batch_train, in zip(range(options.train.epoch.max_nrof_epochs), tf_train_dataset):
            with tf.GradientTape() as tape:
                logits = model(x_batch_train)   # noqa
                loss_value = binary_cross_entropy_loss(logits, options.train)

            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

        logger.info('[{epoch}] Training loss (for one batch) {loss_value}', epoch=epoch, loss_value=loss_value)
        logger.info('variables {model}', model=model)

        conf_mat = ConfusionMatrix(embeddings, model)
        logger.info('ConfusionMatrix \n {conf_mat}', conf_mat=conf_mat)


if __name__ == '__main__':
    main()
