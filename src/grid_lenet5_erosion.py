#!/usr/bin/env cola-env
# -*- coding: utf-8 -*-
"""
Script executes gridsearch in LeDilationNet-5 model.
"""

import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier


from functions.media import read_imgs
from functions.sampler import build_sets
from functions.model import get_lenet5

# Constants
FILE_LOG = '../output/logs/file_log.log'
NEGATIVE_PATH = '/home/gmartins/arquivos/uninove/mestrado/orientacao/projetos/dados/frames/negativos'
POSITIVE_PATH = '/home/gmartins/arquivos/uninove/mestrado/orientacao/projetos/dados/frames/positivos'
FACTOR = .9  # Factor of reduction of image
NUM_CLASSES = 10  # Number of classes
LIST_NUM_FILTERS = [8, 16, 32, 64]  # List of number of filters
LIST_KERNEL_SIZE = [(5, 5), (7, 7), (9, 9)]  # List of filters size
BATCH_SIZE = 32
EPOCHS = 1  # List of epochs
TRAIN_PRCNT = 2 / 3  # Percent considered for training data

# Read images
negative_imgs = read_imgs(NEGATIVE_PATH, FACTOR)
positive_imgs = read_imgs(POSITIVE_PATH, FACTOR)

# Build sets
x_train, y_train, x_test, y_test = build_sets(negative_imgs, positive_imgs, TRAIN_PRCNT)

# Clear lists
negative_imgs, positive_imgs = None, None

# One-hot encode
y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

# Grid search parameters
input_shape = (x_train.shape[1:])
grid_params = dict(num_filters=LIST_NUM_FILTERS, kernel_size=LIST_KERNEL_SIZE)
lenet5_keras = KerasClassifier(build_fn=get_lenet5,
                               input_shape=input_shape,
                               num_classes=NUM_CLASSES,
                               layer_type='Erosion2D',
                               batch_size=BATCH_SIZE,
                               epochs=EPOCHS,
                               verbose=1)
grid = GridSearchCV(estimator=lenet5_keras, param_grid=grid_params, n_jobs=1, cv=10)
grid_result = grid.fit(x_train, y_train)

# Results
print('Best {} accuracy using {}'.format(grid_result.best_score_, grid_result.best_params_))
print('All combinations:')
for acc, params in zip(grid_result.cv_results_['mean_test_score'], grid_result.cv_results_['params']):
    print('Test accuracy {} with {}'.format(acc, params))
