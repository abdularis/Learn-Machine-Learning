# datasetutil.py
# Created by abdularis on 19/11/17

import numpy as np


def unpickle(filename):
    import pickle
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    return data


def load_cifar_batch(filename):
    data = unpickle(filename)

    X = data[b'data']
    Y = data[b'labels']

    X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype('uint8')
    Y = np.array(Y)

    return X, Y


def load_cifar10(folder):
    n_batch = 5

    xs = []
    ys = []

    for n in range(1, n_batch + 1):
        filename = '%s/data_batch_%d' % (folder, n)
        X, Y = load_cifar_batch(filename)
        xs.append(X)
        ys.append(Y)

    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)

    Xte, Yte = load_cifar_batch('%s/test_batch' % folder)
    return Xtr, Ytr, Xte, Yte

