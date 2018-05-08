from os.path import isfile, join, altsep, isdir, basename
from os import listdir, makedirs, errno
from natsort import natsorted, ns
from numba import jit
from matplotlib import pyplot as plt
import numpy as np


def getSequences(path):
    '''
    Auxiliary function that extracts folder names from a given path
    :param path: source path
    :return: array with folder names
    '''
    sequences = [altsep.join((path, f)) for f in listdir(path)
                 if isdir(join(path, f)) and basename(join(path, f)) != 'Results']

    return sequences


def getSamples(path, ext=''):
    '''
    Auxiliary function that extracts file names from a given path based on extension
    :param path: source path
    :param ext: file extension
    :return: array with samples
    '''
    samples = [altsep.join((path, f)) for f in listdir(path)
              if isfile(join(path, f)) and f.endswith(ext)]

    if len(samples) == 0:
        print("ERROR!!! ARRAY OF SAMPLES IS EMPTY (check file extension)")

    return samples


def makeDir(path):
    '''
    To create output path if doesn't exist
    see: https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
    :param path: path to be created
    :return: none
    '''
    try:
        makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

@jit
def natSort(list):
    '''
    Sort frames with human method
    see: https://pypi.python.org/pypi/natsort
    :param list: list that will be sorted
    :return: sorted list
    '''
    return natsorted(list, alg=ns.IGNORECASE)

def plotImages(titles, images, title, row, col):
    fig = plt.figure()
    for i in range(len(images)):
        plt.subplot(row, col, i + 1), plt.imshow(images[i])     #, 'gray'
        if len(titles) != 0:
            plt.title(titles[i])
        plt.gray()
        plt.axis('off')

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    # fig.subplots_adjust(left=0, right=0, top=0, bottom=0)
    plt.show()

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    train_set = [data[i] for i in train_indices]
    test_set  = [data[i] for i in test_indices]
    return train_set, test_set