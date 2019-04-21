from numba import jit
import numpy as np
from sklearn.model_selection import KFold
from scipy import spatial, interpolate
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


def pairwise_distances(embeddings, metric=0):
    if metric == 0:
        # squared Euclidian distance
        dist = spatial.distance.pdist(embeddings, metric='sqeuclidean')
    elif metric == 1:
        # Distance based on cosine similarity
        dist = 1 - spatial.distance.pdist(embeddings, metric='cosine')
        dist = np.arccos(dist) / math.pi
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

    @property
    def accuracy(self):
        return float(self.tp + self.tn) / float(self.tp + self.fp + self.tn + self.fn)

    @property
    def far(self):
        return float(1) if (self.fp + self.tn == 0) else float(self.fp) / float(self.fp + self.tn)

    @property
    def val(self):
        return float(1) if (self.tp + self.fn == 0) else float(self.tp) / float(self.tp + self.fn)

    @property
    def tp_rate(self):
        return float(0) if (self.tp + self.fn == 0) else float(self.tp) / float(self.tp + self.fn)

    @property
    def fp_rate(self):
        return float(0) if (self.fp + self.tn == 0) else float(self.fp) / float(self.fp + self.tn)


def statistics(thresholds, embeddings, labels, far_target=1e-3, nrof_folds=10, distance_metric=0, subtract_mean=False):
    assert (embeddings.shape[0] == len(labels))

    nrof_thresholds = len(thresholds)

    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))

    accuracy = np.zeros(nrof_folds)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    indices = np.arange(len(labels))

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        print('\rROC/VAL {}/{}'.format(fold_idx, nrof_folds), end=utils.end(fold_idx, nrof_folds))

        if subtract_mean:
            mean = np.mean(embeddings[train_set], axis=0)
        else:
            mean = 0.0

        # evaluations with train set
        dist_train = pairwise_distances(embeddings[train_set] - mean, distance_metric)
        labels_train = pairwise_labels(labels[train_set])

        acc_train = np.zeros(nrof_thresholds)
        far_train = np.zeros(nrof_thresholds)

        for idx, threshold in enumerate(thresholds):
            cm = ConfidenceMatrix(threshold, dist_train, labels_train)
            acc_train[idx] = cm.accuracy
            far_train[idx] = cm.far

        # find the best threshold for the fold
        best_threshold_index = np.argmax(acc_train)

        # find the threshold that gives FAR = far_target
        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            far_threshold = f(far_target)
        else:
            far_threshold = 0.0

        # evaluations with test set
        dist_test = pairwise_distances(embeddings[test_set] - mean, distance_metric)
        labels_test = pairwise_labels(labels[test_set])

        for idx, threshold in enumerate(thresholds):
            cm = ConfidenceMatrix(threshold, dist_test, labels_test)
            tprs[fold_idx, idx] = cm.tp_rate
            fprs[fold_idx, idx] = cm.fp_rate

        cm = ConfidenceMatrix(thresholds[best_threshold_index], dist_test, labels_test)
        accuracy[fold_idx] = cm.accuracy

        cm = ConfidenceMatrix(far_threshold, dist_test, labels_test)
        val[fold_idx] = cm.val
        far[fold_idx] = cm.far

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)

    return tpr, fpr, accuracy, val_mean, val_std, far_mean
