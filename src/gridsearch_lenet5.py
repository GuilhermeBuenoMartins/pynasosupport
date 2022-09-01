#!/usr/bin/env cola-env
# -*- coding: utf-8 -*-
"""
Script executes gridsearch in LeNet-5 model.
"""
import logging as log

from tensorflow.python.keras.utils.np_utils import to_categorical

from functions.evaluation import use_model, plot_loss, plot_accuracy
from functions.lenet import get_lenet5
from functions.media import read_imgs
from functions.sampler import build_sets

FILE_LOG = '../output/logs/file_log.log'
NEGATIVE_PATH = '/home/gmartins/arquivos/uninove/mestrado/orientacao/projetos/dados/frames/negativos'
POSITIVE_PATH = '/home/gmartins/arquivos/uninove/mestrado/orientacao/projetos/dados/frames/positivos'
FACTOR = .9  # Factor of reduction of image
NUM_CLASSES = 2  # Number of classes
NUM_FILTERS = 8  # Number of filters
FILTERS_SIZE = 7  # Square filters size
BATCH_SIZE = 32
EPOCHS = 4
LR = .001  # Learning rate
REGULARIZER_PARAM = .002
TRAIN_PRCNT = 2 / 3  # Percent considered for training data

log.basicConfig(filename=FILE_LOG, level=log.INFO)

negative_imgs = read_imgs(NEGATIVE_PATH, FACTOR)
positive_imgs = read_imgs(POSITIVE_PATH, FACTOR)

x_train, y_train, x_test, y_test = build_sets(negative_imgs, positive_imgs, TRAIN_PRCNT)

negative_imgs, positive_imgs = None, None

y_train = to_categorical(y_train, NUM_CLASSES)
y_test = to_categorical(y_test, NUM_CLASSES)

input_shape = (x_train.shape[1:])
model, _ = get_lenet5(input_shape, NUM_CLASSES)
history = use_model(x_train, y_train, x_test, y_test, model, BATCH_SIZE, EPOCHS, LR)

plot_loss(history)
plot_accuracy(history)
