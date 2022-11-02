#!/usr/bin/env cola-env
# -*- coding: utf-8 -*-
"""
Script executes gridsearch in LeDilationNet-5 model.
"""

import logging
import os.path
from datetime import datetime

import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier

from functions.configlog import set_log
from functions.media import read_imgs
from functions.model import get_lenet5
from functions.sampler import build_sets


def main(main_path: str, factor=.9, list_num_filters=[4, 8, 12, 16], list_kernel_size=[(3, 3), (5, 5)], batch_size=None,
         epochs=100, train_prcnt=2 / 3, cpu_mode=False, verbose=1):
    """
    Main function to execution the script.

    :param main_path: Images and logs path.
    :param factor: Factor of images reduction.
    :param list_num_filters: List of number filters.
    :param list_kernel_size: List of kernel size.
    :param batch_size: Batch size value.
    :param epochs: Epochs value.
    :param train_prcnt: Training size percent.
    :return: None
    """

    FILE_NAME = 'grid_lenet5_internal_gradient{}.log'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    NUM_CLASSES = 2
    PATH_NEGATIVO = os.path.join(main_path, 'negativos')
    PATH_POSITIVO = os.path.join(main_path, 'positivos')

    # Setting logs
    set_log(file_name=FILE_NAME, path=main_path)

    # Read images
    negative_imgs = read_imgs(PATH_NEGATIVO, factor)
    positive_imgs = read_imgs(PATH_POSITIVO, factor)

    # Build sets
    x_train, y_train, x_test, y_test = build_sets(negative_imgs, positive_imgs, train_prcnt)

    # Clear lists
    negative_imgs, positive_imgs = None, None

    # One-hot encode
    y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

    # Grid search parameters
    input_shape = (x_train.shape[1:])
    grid_params = dict(num_filters=list_num_filters, kernel_size=list_kernel_size)
    lenet5_keras = KerasClassifier(build_fn=get_lenet5,
                                   input_shape=input_shape,
                                   num_classes=NUM_CLASSES,
                                   layer_type='InternalGradient2D',
                                   batch_size=batch_size,
                                   epochs=epochs,
                                   verbose=verbose,
                                   use_multiprocessing=True
                                   )
    grid = GridSearchCV(estimator=lenet5_keras, param_grid=grid_params, cv=10)
    if cpu_mode:
        with tf.device('/cpu:0'):
            grid_result = grid.fit(x_train, y_train)

    # Results
    logging.info('Best {} accuracy using {}'.format(grid_result.best_score_, grid_result.best_params_))
    logging.info('All combinations:')
    for acc, params in zip(grid_result.cv_results_['mean_test_score'], grid_result.cv_results_['params']):
        logging.info('Test accuracy {} with {}'.format(acc, params))
