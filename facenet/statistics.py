
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import sys

from numba import jit
import os
import datetime
import numpy as np
from skimage import io
from sklearn import metrics
from sklearn.model_selection import KFold
from scipy import spatial, interpolate
from scipy.optimize import brentq
import math

from facenet import utils


@jit(nopython=True)
def pairwise_labels(labels):
    if labels.ndim > 1:
        raise ValueError('label_array: input labels must be 1 dimension')

    nrof_labels = len(labels)

    output = np.zeros(int(nrof_labels*(nrof_labels-1)/2), dtype=np.uint8)
    count = 0

    for i in range(nrof_labels):
        for k in range(i+1, nrof_labels):
            if labels[i] == labels[k]:
                output[count] = 1

            count += 1

    return output


def pairwise_distances(xa, xb=None, metric=0):
    if metric == 0:
        # squared Euclidian distance
        if xb is None:
            dist = spatial.distance.pdist(xa, metric='sqeuclidean')
        else:
            dist = spatial.distance.cdist(xa, xb, metric='sqeuclidean')
    elif metric == 1:
        # Distance based on cosine similarity
        if xb is None:
            dist = spatial.distance.pdist(xa, metric='cosine')
        else:
            dist = spatial.distance.cdist(xa, xb, metric='cosine')
        dist = np.arccos(1 - dist) / math.pi
    else:
        raise 'Undefined distance metric %d' % metric

    return np.array(dist, dtype=np.float32)


@jit(nopython=True)
def confidence_matrix(threshold, distances, labels):

    tp = fp = tn = fn = 0

    for dist, label in zip(distances, labels):
        if dist < threshold:
            if label:
                tp += 1
            else:
                fp += 1
        else:
            if label:
                fn += 1
            else:
                tn += 1

    return tp, fp, tn, fn


class ConfidenceMatrix:
    def __init__(self, threshold, distances, labels):
        self.tp, self.fp, self.tn, self.fn = confidence_matrix(threshold, distances, labels)

        self.tp_rate = float(0) if (self.tp + self.fn == 0) else float(self.tp) / float(self.tp + self.fn)
        self.fp_rate = float(0) if (self.fp + self.tn == 0) else float(self.fp) / float(self.fp + self.tn)
        self.accuracy = float(self.tp + self.tn) / float(self.tp + self.fp + self.tn + self.fn)
        self.far = float(1) if (self.fp + self.tn == 0) else float(self.fp) / float(self.fp + self.tn)
        self.val = float(1) if (self.tp + self.fn == 0) else float(self.tp) / float(self.tp + self.fn)


class Validation:
    def __init__(self, thresholds, embeddings, dbase,
                 far_target=1e-3, nrof_folds=10,
                 distance_metric=0, subtract_mean=False):

        self.dbase = dbase
        self.embeddings = embeddings
        self.subtract_mean = subtract_mean
        self.distance_metric = distance_metric

        labels = dbase.labels
        assert (embeddings.shape[0] == len(labels))

        nrof_thresholds = len(thresholds)

        k_fold = KFold(n_splits=nrof_folds, shuffle=False)

        tprs = np.zeros((nrof_folds, nrof_thresholds))
        fprs = np.zeros((nrof_folds, nrof_thresholds))

        self.best_thresholds = np.zeros(nrof_folds)
        accuracy = np.zeros(nrof_folds)

        val = np.zeros(nrof_folds)
        far = np.zeros(nrof_folds)

        indices = np.arange(len(labels))

        for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
            print('\rvalidation {}/{}'.format(fold_idx, nrof_folds), end=utils.end(fold_idx, nrof_folds))

            if subtract_mean:
                mean = np.mean(embeddings[train_set], axis=0)
            else:
                mean = 0.0

            # evaluations with train set
            dist_train = pairwise_distances(embeddings[train_set] - mean, metric=distance_metric)
            labels_train = pairwise_labels(labels[train_set])

            acc_train = np.zeros(nrof_thresholds)
            far_train = np.zeros(nrof_thresholds)

            for idx, threshold in enumerate(thresholds):
                cm = ConfidenceMatrix(threshold, dist_train, labels_train)
                acc_train[idx] = cm.accuracy
                far_train[idx] = cm.far

            # find the best threshold for the fold
            self.best_thresholds[fold_idx] = thresholds[np.argmax(acc_train)]

            # find the threshold that gives FAR = far_target
            if np.max(far_train) >= far_target:
                f = interpolate.interp1d(far_train, thresholds, kind='slinear')
                far_threshold = f(far_target)
            else:
                far_threshold = 0.0

            # evaluations with test set
            dist_test = pairwise_distances(embeddings[test_set] - mean, metric=distance_metric)
            labels_test = pairwise_labels(labels[test_set])

            for idx, threshold in enumerate(thresholds):
                cm = ConfidenceMatrix(threshold, dist_test, labels_test)
                tprs[fold_idx, idx] = cm.tp_rate
                fprs[fold_idx, idx] = cm.fp_rate

            cm = ConfidenceMatrix(self.best_thresholds[fold_idx], dist_test, labels_test)
            accuracy[fold_idx] = cm.accuracy

            cm = ConfidenceMatrix(far_threshold, dist_test, labels_test)
            val[fold_idx] = cm.val
            far[fold_idx] = cm.far

        self.tp_rate = np.mean(tprs, 0)
        self.fp_rate = np.mean(fprs, 0)

        self.accuracy = accuracy
        self.accuracy_mean = np.mean(self.accuracy)
        self.accuracy_std = np.std(self.accuracy)

        self.val = np.mean(val)
        self.val_std = np.std(val)
        self.far = np.mean(far)
        self.auc = metrics.auc(self.fp_rate, self.tp_rate)

        self.best_threshold = np.mean(self.best_thresholds)
        self.best_threshold_std = np.std(self.best_thresholds)

        try:
            self.eer = brentq(lambda x: 1. - x - interpolate.interp1d(self.fp_rate, self.tp_rate)(x), 0., 1.)
        except Exception:
            self.eer = -1

    def print(self):
        print('Accuracy: {:2.5f}+-{:2.5f}'.format(self.accuracy_mean, self.accuracy_std))
        print('Validation rate: {:2.5f}+-{:2.5f} @ FAR={:2.5f}'.format(self.val, self.val_std, self.far))
        print('Area Under Curve (AUC): {:1.5f}'.format(self.auc))
        print('Equal Error Rate (EER): {:1.5f}'.format(self.eer))
        print('Threshold: {:2.5f}+-{:2.5f}'.format(self.best_threshold, self.best_threshold_std))

    def write_report(self, file, dbase, args):
        git_hash, git_diff = utils.git_hash()
        with open(os.path.expanduser(file), 'at') as f:
            f.write('{}\n'.format(datetime.datetime.now()))
            f.write('git hash: {}\n'.format(git_hash))
            f.write('git diff: {}\n'.format(git_diff))
            f.write('model: {}\n'.format(os.path.expanduser(args.model)))
            f.write('dataset: {}\n'.format(dbase.dirname))
            f.write('number of folders {}\n'.format(dbase.nrof_folders))
            f.write('numbers of images {} and pairs {}\n'.format(dbase.nrof_images, dbase.nrof_pairs))
            f.write('distance metric: {}\n'.format(args.distance_metric))
            f.write('subtract mean: {}\n'.format(args.subtract_mean))
            f.write('Accuracy: {:2.5f}+-{:2.5f}\n'.format(self.accuracy_mean, self.accuracy_std))
            f.write('Validation rate: {:2.5f}+-{:2.5f} @ FAR={:2.5f}\n'.format(self.val, self.val_std, self.far))
            f.write('Area Under Curve (AUC): {:1.5f}\n'.format(self.auc))
            f.write('Equal Error Rate (EER): {:1.5f}\n'.format(self.eer))
            f.write('Threshold: {:2.5f}+-{:2.5f}'.format(self.best_threshold, self.best_threshold_std))
            f.write('\n')


class FalseExamples:
    def __init__(self, dbase, tfrecord, threshold, metric=0, subtract_mean=False):
        self.dbase = dbase
        self.embeddings = tfrecord.embeddings
        self.threshold = threshold
        self.metric = metric
        self.subtract_mean = subtract_mean

    def write_false_pairs(self, fpos_dir, fneg_dir, nrof_images=10):

        if not os.path.isdir(fpos_dir):
            os.makedirs(fpos_dir)

        if not os.path.isdir(fneg_dir):
            os.makedirs(fneg_dir)

        if self.subtract_mean:
            mean = np.mean(self.embeddings, axis=0)
        else:
            mean = 0

        for folder1 in range(self.dbase.nrof_folders):
            print('\rWrite false examples {}/{}'.format(folder1, self.dbase.nrof_folders),
                  end=utils.end(folder1, self.dbase.nrof_folders))

            files1, embeddings1 = self.dbase.extract_data(folder1, self.embeddings)

            # search false negative pairs
            distances = pairwise_distances(embeddings1 - mean, metric=self.metric)
            distances = spatial.distance.squareform(distances)

            for n in range(nrof_images):
                # find maximal distances
                i, k = np.unravel_index(np.argmax(distances), distances.shape)

                if distances[i, k] > self.threshold:
                    self.write_image(distances[i, k], files1[i], files1[k], fneg_dir)
                    distances[[i, k], :] = -1
                    distances[:, [i, k]] = -1
                else:
                    break

            # search false positive pairs
            for folder2 in range(folder1+1, self.dbase.nrof_folders):
                files2, embeddings2 = self.dbase.extract_data(folder2, self.embeddings)

                distances = pairwise_distances(embeddings1-mean, embeddings2-mean, metric=self.metric)

                for n in range(nrof_images//4):
                    # find minimal distances
                    i, k = np.unravel_index(np.argmin(distances), distances.shape)

                    if distances[i, k] < self.threshold:
                        self.write_image(distances[i, k], files1[i], files2[k], fpos_dir)
                        distances[i, :] = np.Inf
                        distances[:, k] = np.Inf
                    else:
                        break

    def generate_filename(self, dirname, distance, file1, file2):
        dir1 = os.path.basename(os.path.dirname(file1))
        name1 = os.path.splitext(os.path.basename(file1))[0]

        dir2 = os.path.basename(os.path.dirname(file2))
        name2 = os.path.splitext(os.path.basename(file2))[0]

        return os.path.join(dirname, '{:2.3f} & {}|{} & {}|{}.png'.format(distance, dir1, name1, dir2, name2))

    def generate_text(self, distance, file1, file2):

        def text(file):
            return os.path.join(os.path.basename(os.path.dirname(file)), os.path.splitext(os.path.basename(file))[0])

        return '{} & {}\n{:2.3f}/{:2.3f}'.format(text(file1), text(file2), distance, self.threshold)

    def write_image(self, distance, file1, file2, dirname, fsize=13):
        fname = self.generate_filename(dirname, distance, file1, file2)
        text = self.generate_text(distance, file1, file2)

        img1 = io.imread(file1)
        img2 = io.imread(file2)
        img = Image.fromarray(np.concatenate([img1, img2], axis=1))

        if sys.platform == 'win32':
            font = ImageFont.truetype("arial.ttf", fsize)
        else:
            font = ImageFont.truetype("LiberationSans-Regular.ttf", fsize)

        draw = ImageDraw.Draw(img)
        draw.text((0, 0), text, (0, 255, 0), font=font)

        img.save(fname)
