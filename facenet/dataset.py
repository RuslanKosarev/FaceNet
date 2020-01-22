
import pathlib
import numpy as np
import math
from facenet import utils, h5utils


class ImageClass:
    """
    Stores the paths to images for a given class
    """
    def __init__(self, name, files, count=None):
        self.name = name
        self.count = count

        self.files = [str(f) for f in files]
        self.files.sort()

    def __str__(self):
        return self.name + ', ' + str(self.nrof_images) + ' images'

    @property
    def nrof_images(self):
        return len(self.files)

    @property
    def nrof_pairs(self):
        return self.nrof_images * (self.nrof_images - 1) // 2

    @property
    def files_as_posix(self):
        return [pathlib.Path(x) for x in self.files]


class DBase:
    def __init__(self, config, extension='', seed=0):
        np.random.seed(seed)

        self.config = config
        self.config.path = pathlib.Path(config.path).expanduser()

        classes = [path for path in self.config.path.glob('*') if path.is_dir()]
        classes.sort()

        if self.config.nrof_classes is not None:
            classes = classes[:self.config.nrof_classes]

        self.classes = []

        for count, path in enumerate(classes):
            files = list(path.glob('*' + extension))
            files.sort()

            if self.config.h5file is not None:
                self.config.h5file = pathlib.Path(self.config.h5file).expanduser()
                files = [f for f in files if h5utils.read(self.config.h5file, h5utils.filename2key(f, 'is_valid'), default=True)]

            if self.config.nrof_images is not None:
                if len(files) > self.config.nrof_images:
                    files = np.random.choice(files, size=self.config.nrof_images, replace=False)

            if len(files) > 0:
                self.classes.append(ImageClass(path.stem, files, count=count))
                print('\r({}/{}) class {}'.format(count, len(classes), self.classes[-1].name),
                      end=utils.end(count, len(classes)))

    @property
    def labels(self):
        labels = []
        for idx, cls in enumerate(self.classes):
            labels += [idx] * cls.nrof_images
        return np.array(labels)

    def __repr__(self):
        """Representation of the database"""
        info = ('class {}\n'.format(self.__class__.__name__) +
                'Directory to load images {}\n'.format(self.config.path) +
                'h5 file to filter images {}\n'.format(self.config.h5file) +
                'Number of classes {} \n'.format(self.nrof_classes) +
                'Number of images {}\n'.format(self.nrof_images) +
                'Number of pairs {}\n'.format(self.nrof_pairs) +
                'Number of positive pairs {} ({:.6f} %)\n'.
                format(self.nrof_positive_pairs, 100 * self.nrof_positive_pairs / self.nrof_pairs) +
                'Number of negative pairs {} ({:.6f} %)\n'.
                format(self.nrof_negative_pairs, 100 * self.nrof_negative_pairs / self.nrof_pairs))
        return info

    @property
    def nrof_classes(self):
        return len(self.classes)

    @property
    def nrof_images(self):
        return sum(cls.nrof_images for cls in self.classes)

    @property
    def nrof_negative_pairs(self):
        return self.nrof_pairs - self.nrof_positive_pairs

    @property
    def nrof_positive_pairs(self):
        return sum(cls.nrof_pairs for cls in self.classes)

    @property
    def nrof_pairs(self):
        return self.nrof_images * (self.nrof_images - 1) // 2

    @property
    def files(self):
        f = []
        for cls in self.classes:
            f += cls.files
        return f

    @property
    def files_as_posix(self):
        f = []
        for cls in self.classes:
            f += cls.files_as_posix
        return f

    def extract_data(self, folder_idx, embeddings=None):
        indices = np.where(self.labels == folder_idx)[0]
        files = [self.files[idx] for idx in indices]

        if embeddings is None:
            return files
        else:
            return files, embeddings[indices]

    def split(self, split_ratio, min_nrof_images_per_class, mode='images'):
        if split_ratio <= 0.0:
            return self.classes, []

        if mode == 'classes':
            nrof_classes = len(self.classes)
            class_indices = np.arange(nrof_classes)
            np.random.shuffle(class_indices)
            split = int(round(nrof_classes * (1 - split_ratio)))
            train_set = [self.classes[i] for i in class_indices[0:split]]
            test_set = [self.classes[i] for i in class_indices[split:-1]]
        elif mode == 'images':
            train_set = []
            test_set = []
            for cls in self.classes:
                paths = cls.files
                np.random.shuffle(paths)
                nrof_images_in_class = len(paths)
                split = int(math.floor(nrof_images_in_class * (1 - split_ratio)))
                if split == nrof_images_in_class:
                    split = nrof_images_in_class - 1
                if split >= min_nrof_images_per_class and nrof_images_in_class - split >= 1:
                    train_set.append(ImageClass(cls.name, paths[:split]))
                    test_set.append(ImageClass(cls.name, paths[split:]))
        else:
            raise ValueError('Invalid train/test split mode "%s"' % mode)

        return train_set, test_set
