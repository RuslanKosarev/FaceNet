# coding:utf-8
__author__ = 'Ruslan N. Kosarev'

import os
import time
import numpy as np
import pathlib as plib
import datetime
import tensorflow as tf
from PIL import Image
from subprocess import Popen, PIPE
import facenet
from facenet.config import src_dir


def end(start, stop):
    return '\n' if (start+1) == stop else ''


def store_revision_info(output_filename, arg_string, mode='w'):
    output_filename = plib.Path(output_filename)

    if output_filename.is_dir():
        output_filename = output_filename.joinpath(output_filename, 'revision_info.txt')

    arg_string = ' '.join(arg_string)

    try:
        # Get git hash
        cmd = ['git', 'rev-parse', 'HEAD']
        gitproc = Popen(cmd, stdout=PIPE, cwd=str(src_dir))
        (stdout, _) = gitproc.communicate()
        git_hash = stdout.strip()
    except OSError as e:
        git_hash = ' '.join(cmd) + ': ' + e.strerror

    try:
        # Get local changes
        cmd = ['git', 'diff', 'HEAD']
        gitproc = Popen(cmd, stdout=PIPE, cwd=str(src_dir))
        (stdout, _) = gitproc.communicate()
        git_diff = stdout.strip()
    except OSError as e:
        git_diff = ' '.join(cmd) + ': ' + e.strerror

    # Store a text file in the log directory
    with open(str(output_filename), mode) as f:
        f.write('{}\n'.format(datetime.datetime.now()))
        f.write('arguments: %s\n--------------------\n' % arg_string)
        f.write('tensorflow version: %s\n--------------------\n' % tf.__version__)  # @UndefinedVariable
        f.write('git hash: %s\n--------------------\n' % git_hash)
        f.write('git diff: %s\n' % git_diff)
        f.write('\n')


def write_arguments(arguments, filename):
    makedirs(filename.parent)

    with open(str(filename), 'w') as f:
        def write_to_file(dct, ident=''):
            shift = 3 * ' '

            for key, item in dct.items():
                if isinstance(item, facenet.config.YAMLConfig):
                    f.write('{}{}:\n'.format(ident, key))
                    write_to_file(item, ident=ident + shift)
                else:
                    f.write('{}{}: {}\n'.format(ident, key, str(item)))

        write_to_file(arguments)


def makedirs(dir_name):
    dir_name = plib.Path(dir_name).expanduser()
    if not dir_name.exists():
        dir_name.mkdir(parents=True)


def write_image(image, filename, prefix=None, mode='RGB'):
    if prefix is not None:
        filename = plib.Path(prefix).joinpath(filename)
    filename = plib.Path(filename).expanduser()

    if isinstance(image, np.ndarray):
        image = array2pil(image, mode=mode)
    else:
        # to avoid some warnings while tf reads saved files
        image = array2pil(pil2array(image))

    if image.save(str(filename)):
        raise IOError('while writing the file {}'.format(filename))


def read_image(filename, prefix=None, mode=None):
    if prefix is not None:
        image = Image.open(os.path.join(str(prefix), str(filename)))
    else:
        image = Image.open(str(filename))

    if mode is not None:
        image = image.convert(mode)

    return image


class ImageLoader:
    def __iter__(self):
        return self

    def __init__(self, input, prefix=None, display=100, log=True):

        if not isinstance(input, (str, list)):
            raise IOError('Input \'{}\' must be directory or list of files'.format(input))

        if isinstance(input, list):
            self.files = input
        elif os.path.isdir(os.path.expanduser(input)):
            prefix = os.path.expanduser(input)
            self.files = os.listdir(prefix)
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
            return image
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

