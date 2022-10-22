"""
Functions for easy handling of training and test dataset.
"""

import logging as log
import numpy as np
from sklearn.model_selection import train_test_split


def build_sets(negative_imgs, positive_imgs, train_prcnt, verbose=True):
    """
    The function build a training and test sets from negative and positive images lists.

    :param negative_imgs: list of negative images.
    :param positive_imgs: list of positive images.
    :param train_prcnt: percent of samples intended for train.
    :param verbose: print in the terminal
    :return: x_train, y_train, x_test, y_test
    """

    m_negative = len(negative_imgs)
    m_positive = len(positive_imgs)
    x_set = np.concatenate((np.array(negative_imgs), np.array(positive_imgs)))
    x_set = x_set.astype('float32') / 255
    y_set = np.concatenate((np.zeros(m_negative), np.ones(m_positive)))

    m = m_negative + m_positive
    threshold = round(m * train_prcnt)

    x_train, x_test, y_train, y_test = train_test_split(x_set, y_set, train_size=threshold, stratify=y_set, random_state=33)

    msg = '\nBuild sets\n' \
          '================================\n' \
          'Training samples shape:\t {}\n' \
          'Test samples shape:\t {} \n' \
          '--------------------------\n' \
          'Total of samples:\t {}\n'

    if verbose:
        print(msg.format(x_train.shape, x_test.shape, m))
    log.info(msg.format(x_train.shape, x_test.shape, m))

    return x_train, y_train, x_test, y_test
