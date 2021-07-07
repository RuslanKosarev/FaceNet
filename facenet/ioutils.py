# coding:utf-8
__author__ = 'Ruslan N. Kosarev'

import os
import sys
import platform
import time
import numpy as np
from functools import partial
from pathlib import Path
import datetime
import tensorflow as tf
from PIL import Image
from subprocess import Popen, PIPE
from facenet import config, h5utils


makedirs = partial(Path.mkdir, parents=True, exist_ok=True)


def end(start, stop):
    return '\n' if (start+1) == stop else ''


def get_time():
    return time.monotonic()


def write_elapsed_time(files, start_time):
    if not isinstance(files, list):
        files = [files]

    for file in files:
        file = Path(file).expanduser()
        elapsed_time = (time.monotonic() - start_time)/60

        if file.suffix == '.h5':
            h5utils.write(file, 'elapsed_time', elapsed_time)
        else:
            with file.open('at') as f:
                f.write('elapsed time: {:.3f}\n'.format(elapsed_time))


def store_revision_info(output_filename, mode='w'):
    output_filename = Path(output_filename)

    if output_filename.is_dir():
        output_filename = output_filename.joinpath(output_filename, 'revision_info.txt')

    arg_string = ' '.join(sys.argv)

    git_hash_ = git_hash()
    git_diff_ = git_diff()

    # Store a text file in the log directory
    with open(str(output_filename), mode) as f:
        f.write(64 * '-' + '\n')
        f.write('{} {}\n'.format('store_revision_info', datetime.datetime.now()))
        f.write('release version: {}\n'.format(platform.version()))
        f.write('python version: {}\n'.format(sys.version))
        f.write('tensorflow version: {}\n'.format(tf.__version__))
        f.write('arguments: {}\n'.format(arg_string))
        f.write('git hash: {}\n'.format(git_hash_))
        f.write('git diff: {}\n'.format(git_diff_))
        f.write('\n')


def git_hash():
    src_path, _ = os.path.split(os.path.realpath(__file__))

    try:
        # Get git hash
        cmd = ['git', 'rev-parse', 'HEAD']
        gitproc = Popen(cmd, stdout=PIPE, cwd=src_path)
        (stdout, _) = gitproc.communicate()
        info = stdout.strip()
    except OSError as e:
        info = ' '.join(cmd) + ': ' + e.strerror

    return info


def git_diff():
    src_path, _ = os.path.split(os.path.realpath(__file__))

    try:
        # Get local changes
        cmd = ['git', 'diff', 'HEAD']
        gitproc = Popen(cmd, stdout=PIPE, cwd=src_path)
        (stdout, _) = gitproc.communicate()
        info = stdout.strip()
    except OSError as e:
        info = ' '.join(cmd) + ': ' + e.strerror

    return info


def write_arguments(cfg, path, mode='w'):
    path = Path(path).expanduser()
    name = Path(sys.argv[0]).stem + '.yaml'

    file = path / name

    makedirs(file.parent)

    with file.open(mode=mode) as f:
        f.write('{}\n'.format(str(cfg)))


def write_image(image, filename, prefix=None, mode='RGB'):
    if prefix is not None:
        filename = Path(prefix).joinpath(filename)
    filename = Path(filename).expanduser()

    if isinstance(image, np.ndarray):
        image = array2pil(image, mode=mode)
    else:
        # to avoid some warnings while tf reads saved files
        image = array2pil(pil2array(image))

    if image.save(str(filename)):
        raise IOError('while writing the file {}'.format(filename))


def read_image(file, prefix=None):
    file = Path(file)
    if prefix is not None:
        file = Path(prefix).joinpath(file)

    image = Image.open(file)
    if image is None:
        raise IOError('while reading the file {}'.format(file))

    return image


class ImageLoader:
    def __iter__(self):
        return self

    def __init__(self, input, prefix=None, display=100, log=True):

        if not isinstance(input, (Path, list)):
            raise IOError('Input \'{}\' must be directory or list of files'.format(input))

        if isinstance(input, list):
            self.files = input
        elif input.is_dir():
            prefix = input.expanduser()
            self.files = list(prefix.glob('*'))
        else:
            raise IOError('Directory \'{}\' does not exist'.format(input))

        self.counter = 0
        self.start_time = time.time()
        self.display = display
        self.size = len(self.files)
        self.prefix = str(prefix)
        self.log = log
        self.__filename = None

        print('Loader <{}> is initialized, number of files {}'.format(self.__class__.__name__, self.size))

    def __next__(self):
        if self.counter < self.size:
            if (self.counter + 1) % self.display == 0:
                elapsed_time = (time.time() - self.start_time) / self.display
                print('\rnumber of processed images {}/{}, {:.5f} seconds per image'.format(self.counter+1, self.size, elapsed_time), end='')
                self.start_time = time.time()

            image = read_image(self.files[self.counter], prefix=self.prefix)
            self.filename = image.filename

            if self.log:
                print('{}/{}, {}, {}'.format(self.counter, self.size, self.filename, image.size))

            self.counter += 1
            return pil2array(image)
        else:
            print('\n\rnumber of processed images {}'.format(self.size))
            raise StopIteration

    def reset(self):
        self.counter = 0
        return self


def pil2array(image, mode='RGB'):
    return np.array(image.convert(mode.upper()))


def array2pil(image, mode='RGB'):

    default_mode = 'RGB'
    index = []

    for sym in mode.upper():
        index.append(default_mode.index(sym))

    output = Image.fromarray(image[:, :, index], mode=default_mode)

    return output


def write_to_file(file, s, mode='w'):
    file = Path(file).expanduser()
    with file.open(mode=mode) as f:
        f.write(s)


def write_text_log(file, info):
    info_str = 64 * '-' + '\n' + str(info)

    if info_str[-1] != '\n':
        info_str += '\n'

    with file.open(mode='a') as f:
        f.write(info_str)


def glob_single_file(model_dir, pattern):
    files = list(model_dir.glob(pattern))

    if len(files) != 1:
        raise ValueError(f'There should not be more than single files in the model directory {model_dir}.')

    return files[0]

