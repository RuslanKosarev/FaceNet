"""Performs processing images."""
# MIT License
#
# Copyright (c) 2019 Ruslan N. Kosarev

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import sys
from tqdm import tqdm

import time
import datetime
import numpy as np
from collections import Iterable
import sklearn
from sklearn.model_selection import KFold
from scipy import spatial, interpolate
from scipy.optimize import brentq
from pathlib import Path

from facenet import utils, ioutils, h5utils


def pairwise_similarities(xa, xb=None, metric=0, atol=1.e-5):
    """
    Evaluate pairwise distances between vectors xa and xb
    :param xa:
    :param xb:
    :param metric: 0 --- distance or 1 --- cosine distance
    :param atol:
    :return:
    """

    if xb is None:
        sims = xa @ xa.transpose()
        sims = sims[np.triu_indices(sims.shape[0], k=1)]
    else:
        sims = xa @ xb.transpose()

    if sims.size > 0:
        # embeddings in xa, xb must be normalized to 1, and therefore sims must be in range (-1, +1)
        lim = 1 + atol
        if sims.min() < -lim or sims.max() > lim:
            raise ValueError('\nembeddings must be normalized to 1, range {} {}'.format(sims.min(), sims.max()))

        # to avoid warnings in np.arccos()
        sims[sims < -1] = -1
        sims[sims > +1] = +1

        if metric == 0:
            # squared Euclidean distance
            sims = 2 * (1 - sims)
        elif metric == 1:
            # cosine
            sims = np.arccos(sims)
        else:
            raise ValueError('Undefined similarity metric {}'.format(metric))

    return sims


def mean(x):
    return np.mean(np.array(x))


def std(x):
    return np.std(np.array(x))


def split_embeddings(embeddings, labels):
    """
    split embeddings to structure [[], [], ...[]]
    :param embeddings:
    :param labels:
    :return:
    """
    emb_list = []
    for label in np.unique(labels):
        emb_array = embeddings[label == labels]
        emb_list.append(emb_array)
    return emb_list


class SimilarityCalculator:
    """
    Class to evaluate similarities according to defined metric
    """
    def __init__(self, embeddings, labels, metric=0):
        self.metric = metric
        self.embeddings = split_embeddings(embeddings, labels)

    def evaluate(self, i, k):
        nrof_positive_class_pairs = self.nrof_classes
        nrof_negative_class_pairs = self.nrof_classes * (self.nrof_classes - 1) / 2

        if i == k:
            sims = pairwise_similarities(self.embeddings[i], metric=self.metric)
            weight = sims.size * nrof_positive_class_pairs
        else:
            sims = pairwise_similarities(self.embeddings[i], self.embeddings[k], metric=self.metric)
            weight = sims.size * nrof_negative_class_pairs

        return sims, weight

    @property
    def nrof_classes(self):
        return len(self.embeddings)

    def nrof_images(self, i):
        return self.embeddings[i].shape[0]


class ConfidenceMatrix:
    """
    Class to evaluate confidence matrix (tp, tn, fp, fn) and others metrics
    """
    def __init__(self, calculator, threshold):

        self.threshold = np.array(threshold, ndmin=1)

        self.tp = np.zeros(self.threshold.size)
        self.tn = np.zeros(self.threshold.size)
        self.fp = np.zeros(self.threshold.size)
        self.fn = np.zeros(self.threshold.size)

        for i in range(calculator.nrof_classes):
            for k in range(i+1):
                sims, weight = calculator.evaluate(i, k)
                if sims.size < 1:
                    continue

                for n, threshold in enumerate(self.threshold):
                    count = np.count_nonzero(sims < threshold)

                    if i == k:
                        self.tp[n] += count/weight
                        self.fn[n] += (sims.size - count)/weight
                    else:
                        self.fp[n] += count/weight
                        self.tn[n] += (sims.size - count)/weight

    @property
    def accuracy(self):
        return (self.tp + self.tn) / (self.tp + self.fp + self.tn + self.fn)

    @property
    def precision(self):
        i = (self.tp + self.fp) > 0
        precision = np.ones(self.threshold.size)
        precision[i] = self.tp[i] / (self.tp[i] + self.fp[i])
        return precision

    @property
    def tp_rates(self):
        # true positive rate, validation rate, sensitivity or recall
        i = (self.tp + self.fn) > 0
        tp_rates = np.ones(self.threshold.size)
        tp_rates[i] = self.tp[i] / (self.tp[i] + self.fn[i])
        return tp_rates

    @property
    def tn_rates(self):
        # true negative rate, 1 - false alarm rate, specificity
        i = (self.tn + self.fp) > 0
        tn_rates = np.ones(self.threshold.size)
        tn_rates[i] = self.tn[i] / (self.tn[i] + self.fp[i])
        return tn_rates

    @property
    def fp_rates(self):
        # false positive rate, false alarm rate
        return 1 - self.tn_rates

    @property
    def fn_rates(self):
        # false negative rate,
        return 1 - self.tp_rates


class Report:
    """
    Class to generate statistical report
    """
    def __init__(self, criterion=None):
        self.criterion = criterion
        self.conf_matrix_train = []
        self.conf_matrix_test = []

    def __repr__(self):
        dct = self.dict

        info = self.criterion + '\n'

        info += ('Area under curve (AUC): {:1.5f}\n'.format(dct['auc']) +
                 'Equal error rate (EER): {:1.5f}\n'.format(dct['eer']) + '\n'
                 )

        info += ('Accuracy:  {:2.5f}+-{:2.5f}\n'.format(dct['accuracy'], dct['accuracy_std']) +
                 'Precision: {:2.5f}+-{:2.5f}\n'.format(dct['precision'], std(dct['precision_std'])) +
                 'Sensitivity (TPR, 1-a type 1 error): {:2.5f}+-{:2.5f}\n'.format(dct['tp_rates'], dct['tp_rates_std']) +
                 'Specificity (TNR, 1-b type 2 error): {:2.5f}+-{:2.5f}\n'.format(dct['tn_rates'], dct['tn_rates_std']) +
                 'Threshold: {:2.5f}+-{:2.5f}\n'.format(dct['threshold'], dct['threshold_std']) + '\n'
                 )
        return info

    def append_fold(self, name, conf_matrix):
        if name == 'train':
            self.conf_matrix_train.append(conf_matrix)
        else:
            self.conf_matrix_test.append(conf_matrix)

    @property
    def dict(self):
        tp_rates = np.mean(np.array([m.tp_rates for m in self.conf_matrix_train]), axis=0)
        tn_rates = np.mean(np.array([m.tn_rates for m in self.conf_matrix_train]), axis=0)

        dct = {'auc': -1, 'eer': -1}
        try:
            dct['auc'] = sklearn.metrics.auc(1 - tn_rates, tp_rates)
        except:
            pass

        try:
            dct['eer'] = brentq(lambda x: 1. - x - interpolate.interp1d(1 - tn_rates, tp_rates)(x), 0., 1.)
        except:
            pass

        def get(name):
            return [m.__getattribute__(name) for m in self.conf_matrix_test]

        for key in ('accuracy', 'precision', 'tp_rates', 'tn_rates', 'threshold'):
            x = get(key)
            dct[key] = np.mean(x)
            dct[key + '_std'] = np.std(x)

        return dct


class FaceToFaceValidation:
    """
    Class to perform face-to-face validation
    """
    def __init__(self, embeddings, labels, config, info=''):
        """
        :param embeddings:
        :param labels:
        """
        self.elapsed_time = time.monotonic()
        self.embeddings = embeddings
        self.labels = labels
        self.info = info

        assert (embeddings.shape[0] == len(labels))

        self.config = config
        self.reports = None

        if self.config.metric == 0:
            upper_threshold = 4
        elif self.config.metric == 1:
            upper_threshold = np.pi
        else:
            raise ValueError('Undefined similarity metric {}'.format(self.config.metric))

        self.thresholds = np.linspace(0, upper_threshold, 100)

        self._evaluate()

    def __repr__(self):
        """Representation of the database"""
        info = ('{} {}\n'.format(self.__class__.__name__, self.info) +
                'metric: {}\n\n'.format(self.config.metric))
        for r in self.reports:
            info += str(r)
        info += 'elapsed_time: {}\n'.format(self.elapsed_time)
        return info

    def _evaluate(self):
        k_fold = KFold(n_splits=self.config.nrof_folds, shuffle=True, random_state=0)
        indices = np.arange(len(self.labels))

        self.reports = (
            Report(criterion='MaximumAccuracy'),
            Report(criterion='FalseAlarmRate(FAR = {})'.format(self.config.far_target))
        )

        with tqdm(total=k_fold.n_splits) as bar:
            for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
                # evaluations with train set and define the best threshold for the fold
                calculator = SimilarityCalculator(self.embeddings[train_set], self.labels[train_set], metric=self.config.metric)

                matrix = ConfidenceMatrix(calculator, self.thresholds)
                for i in range(len(self.reports)):
                    self.reports[i].append_fold('train', matrix)

                # find the threshold that gives maximal accuracy
                accuracy_threshold = self.thresholds[np.argmax(matrix.accuracy)]

                # find the threshold that gives FAR (FPR, 1-TNR) = far_target
                far_threshold = 0
                if np.max(matrix.fp_rates) >= self.config.far_target:
                    f = interpolate.interp1d(matrix.fp_rates, self.thresholds, kind='slinear')
                    far_threshold = f(self.config.far_target)

                # evaluations with test set
                calculator = SimilarityCalculator(self.embeddings[test_set], self.labels[test_set], metric=self.config.metric)

                self.reports[0].append_fold('test', ConfidenceMatrix(calculator, accuracy_threshold))
                self.reports[1].append_fold('test', ConfidenceMatrix(calculator, far_threshold))

                bar.set_postfix_str(self.__class__.__name__)
                bar.update()

        self.elapsed_time = time.monotonic() - self.elapsed_time

    @property
    def dict(self):
        output = {r.criterion: r.dict for r in self.reports}
        return output

    def write_report(self, file):
        file = Path(file).expanduser()

        with file.open('at') as f:
            f.write(64 * '-' + '\n')
            f.write('{} {}\n'.format(self.__class__.__name__, datetime.datetime.now()))
            f.write('metric: {}\n\n'.format(self.config.metric))
            for r in self.reports:
                f.write(str(r))

    def write_h5file(self, h5file, tag=None):
        h5utils.write_dict(h5file, self.dict, group=tag)


# class FalseExamples:
#     def __init__(self, dbase, tfrecord, threshold, metric=0, subtract_mean=False):
#         self.dbase = dbase
#         self.embeddings = tfrecord.data
#         self.threshold = threshold
#         self.metric = metric
#         self.subtract_mean = subtract_mean
#
#     def write_false_pairs(self, fpos_dir, fneg_dir, nrof_fpos_images=10, nrof_fneg_images=2):
#         ioutils.makedirs(fpos_dir)
#         ioutils.makedirs(fneg_dir)
#
#         if self.subtract_mean:
#             mean = np.mean(self.embeddings, axis=0)
#         else:
#             mean = 0
#
#         for folder1 in range(self.dbase.nrof_folders):
#             print('\rWrite false examples {}/{}'.format(folder1, self.dbase.nrof_folders),
#                   end=utils.end(folder1, self.dbase.nrof_folders))
#
#             files1, embeddings1 = self.dbase.extract_data(folder1, self.embeddings)
#
#             # search false negative pairs
#             similarities = pairwise_similarities(embeddings1 - mean, metric=self.metric)
#             similarities = spatial.distance.squareform(similarities)
#
#             for n in range(nrof_fpos_images):
#                 # find maximal distances
#                 i, k = np.unravel_index(np.argmax(similarities), similarities.shape)
#
#                 if similarities[i, k] > self.threshold:
#                     self.write_image(similarities[i, k], files1[i], files1[k], fneg_dir)
#                     similarities[[i, k], :] = -1
#                     similarities[:, [i, k]] = -1
#                 else:
#                     break
#
#             # search false positive pairs
#             for folder2 in range(folder1+1, self.dbase.nrof_folders):
#                 files2, embeddings2 = self.dbase.extract_data(folder2, self.embeddings)
#
#                 similarities = pairwise_similarities(embeddings1 - mean, embeddings2 - mean, metric=self.metric)
#
#                 for n in range(nrof_fneg_images):
#                     # find minimal distances
#                     i, k = np.unravel_index(np.argmin(similarities), similarities.shape)
#
#                     if similarities[i, k] < self.threshold:
#                         self.write_image(similarities[i, k], files1[i], files2[k], fpos_dir)
#                         similarities[i, :] = np.Inf
#                         similarities[:, k] = np.Inf
#                     else:
#                         break
#
#     def generate_filename(self, dirname, distance, file1, file2):
#         dir1 = os.path.basename(os.path.dirname(file1))
#         name1 = os.path.splitext(os.path.basename(file1))[0]
#
#         dir2 = os.path.basename(os.path.dirname(file2))
#         name2 = os.path.splitext(os.path.basename(file2))[0]
#
#         return os.path.join(dirname, '{:2.3f} & {}|{} & {}|{}.png'.format(distance, dir1, name1, dir2, name2))
#
#     def generate_text(self, distance, file1, file2):
#
#         def text(file):
#             return os.path.join(os.path.basename(os.path.dirname(file)), os.path.splitext(os.path.basename(file))[0])
#
#         return '{} & {}\n{:2.3f}/{:2.3f}'.format(text(file1), text(file2), distance, self.threshold)
#
#     def write_image(self, distance, file1, file2, dirname, fsize=13):
#         fname = self.generate_filename(dirname, distance, file1, file2)
#         text = self.generate_text(distance, file1, file2)
#
#         img1 = io.imread(file1)
#         img2 = io.imread(file2)
#         img = Image.fromarray(np.concatenate([img1, img2], axis=1))
#
#         if sys.platform == 'win32':
#             font = ImageFont.truetype("arial.ttf", fsize)
#         else:
#             font = ImageFont.truetype("LiberationSans-Regular.ttf", fsize)
#
#         draw = ImageDraw.Draw(img)
#         draw.text((0, 0), text, (0, 255, 0), font=font)
#
#         img.save(fname)
