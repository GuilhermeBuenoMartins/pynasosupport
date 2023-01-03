#!/usr/bin/env cola-env
# -*- coding: utf-8 -*-
"""
Script of ResNet34 execution
"""
import logging
import os.path
from datetime import datetime

import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier

from functions.configlog import set_log
from functions.media import read_imgs
from functions.model import get_renet34
from functions.sampler import build_sets


def main(main_path: str, factor=.9, layer_type='Conv2D', list_num_filters=[4, 8, 12, 16], list_kernel_size=[(3, 3), (5, 5)], batch_size=None,
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

    FILE_NAME = 'grid_resnet34_{}{}.log'.format(layer_type, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
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
    lenet5_keras = KerasClassifier(build_fn=get_renet34,
                                   input_shape=input_shape,
                                   num_classes=NUM_CLASSES,
                                   layer_type=layer_type,
                                   batch_size=batch_size,
                                   epochs=epochs,
                                   verbose=verbose,
                                   use_multiprocessing=True
                                   )
    grid = GridSearchCV(estimator=lenet5_keras, param_grid=grid_params, cv=10)
    if cpu_mode:
        with tf.device('/cpu:0'):
            grid_result = grid.fit(x_train, y_train, validation_data=(x_test, y_test))
    else:
        grid_result = grid.fit(x_train, y_train, validation_data=(x_test, y_test))

    # Results
    logging.info('Best {} accuracy using {}'.format(grid_result.best_score_, grid_result.best_params_))
    logging.info('All combinations:')
    for acc, params in zip(grid_result.cv_results_['mean_test_score'], grid_result.cv_results_['params']):
        logging.info('Test accuracy {} with {}'.format(acc, params))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Create a Grid Lenet-5 Convolution')
    parser.add_argument('--main_path', type=str, required=True, help='Images and logs path.')
    parser.add_argument('--factor', type=float, required=False, default=.9, help='Factor of images reduction.')
    parser.add_argument('--list_num_filters', type=int, nargs='*', required=False, default=[4, 8, 12, 16], help='List of number filters.')
    parser.add_argument('--list_kernel_size', type=str, nargs='*', required=False, default=["3,3", "5,5"], help='List of kernel size.')
    parser.add_argument('--batch_size', type=int, required=False, default=None, help='Batch size value.')
    parser.add_argument('--epochs', type=int, required=False, default=100, help='Epochs value.')
    parser.add_argument('--train_prcnt', type=float, required=False, default=2/3, help='Training size percent.')
    parser.add_argument('--cpu_mode', type=bool, required=False, default=False, help='When true give preference to CPU use.')
    parser.add_argument('--verbose', type=bool, required=False, default=1, help='When equals 1, print trainning on screen.')
    args = parser.parse_args()
    args_kernel_size = [eval(kernel_size) for kernel_size in args.list_kernel_size]
    main(main_path=args.main_path, factor=args.factor, list_num_filters=args.list_num_filters,
         list_kernel_size=args_kernel_size, batch_size=args.batch_size, epochs=args.epochs,
         train_prcnt=args.train_prcnt, cpu_mode=args.cpu_mode, verbose=args.verbose)