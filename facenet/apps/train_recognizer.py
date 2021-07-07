"""Train facenet classifier.
"""
# MIT License
#
# Copyright (c) 2020 SMedX

import click
from tqdm import tqdm
from pathlib import Path
from loguru import logger

import tensorflow as tf
import numpy as np

from facenet import config, facenet, faceclass, ioutils, h5utils


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
        return (f'{self.__class__.__name__}\n' +
                f'{str(self.classifier)}\n' +
                f'accuracy  {self.accuracy}\n' +
                f'precision {self.precision}\n' +
                f'tp rate   {self.tp_rate}\n' +
                f'tn rate   {self.tn_rate}\n')


def binary_cross_entropy_loss(logits, options):
    # define upper-triangle indices
    batch_size = options.nrof_classes_per_batch * options.nrof_examples_per_class
    triu_indices = [(i, k) for i, k in zip(*np.triu_indices(batch_size, k=1))]

    # compute labels for embeddings
    labels = []
    for i, k in triu_indices:
        if (i // options.nrof_examples_per_class) == (k // options.nrof_examples_per_class):
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

    cfg = config.train_recognizer(model)

    h5file = list(model.glob('*.h5'))[0]
    embeddings = h5utils.read(h5file, 'embeddings')
    labels = h5utils.read(h5file, 'labels')

    embeddings = facenet.split_embeddings(embeddings, labels)

    cfg.recognizer.nrof_classes_per_batch = 8
    cfg.recognizer.nrof_examples_per_class = 8
    tf_train_dataset = facenet.equal_batches_input_pipeline(embeddings, cfg.recognizer)

    # define classifier
    if cfg.embeddings.normalize:
        model = faceclass.FaceToFaceNormalizedEmbeddingsClassifier()
    else:
        model = faceclass.FaceToFaceDistanceClassifier()

    logits = model(embeddings_batch)
    cross_entropy = binary_cross_entropy_loss(logits, cfg)

    # define train operations
    global_step = tf.Variable(0, trainable=False, name='global_step')

    dtype = tf.float64
    initial_learning_rate = tf.constant(cfg.train.learning_rate_schedule.initial_value, dtype=dtype)
    decay_rate = tf.constant(cfg.train.learning_rate_schedule.decay_rate, dtype=dtype)

    if not cfg.train.learning_rate_schedule.decay_steps:
        decay_steps = tf.constant(cfg.train.epoch.size, dtype=dtype)
    else:
        decay_steps = tf.constant(cfg.train.learning_rate_schedule.decay_steps, dtype=dtype)

    lr_decay_factor = tf.math.pow(decay_rate, tf.math.floor(tf.cast(global_step, dtype=dtype) / decay_steps))
    learning_rate = initial_learning_rate * lr_decay_factor

    train_ops = facenet.train_op(cfg.train, cross_entropy, global_step, learning_rate, tf.global_variables())

    tensor_ops = {
        'global_step': global_step,
        'loss': cross_entropy,
        'vars': tf.trainable_variables(),
        'learning_rate': learning_rate
    }

    print('start training')

    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        for epoch in range(cfg.train.epoch.max_nrof_epochs):
            with tqdm(total=cfg.train.epoch.size) as bar:
                for _ in range(cfg.train.epoch.size):
                    embeddings_batch_np = session.run(next_elem)
                    feed_dict = {embeddings_batch: embeddings_batch_np}

                    _, outs = session.run([train_ops, tensor_ops], feed_dict=feed_dict)

                    postfix = f"variables {outs['vars']}, loss {outs['loss']}"
                    bar.set_postfix_str(postfix)
                    bar.update()

            info = f"epoch [{epoch + 1}/{cfg.train.epoch.max_nrof_epochs}], learning rate {outs['learning_rate']}"
            print(info)

            conf_mat = ConfusionMatrix(embarray, model)
            print(conf_mat)
            ioutils.write_text_log(cfg.logfile, info)
            ioutils.write_text_log(cfg.logfile, conf_mat)

    print(f'Model has been saved to the directory: {cfg.classifier.path}')


if __name__ == '__main__':
    main()
