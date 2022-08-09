#!/usr/bin/env cola-env
# -*- coding: utf-8 -*-
"""
Script of layers comparison between morpholayers and conv-layers
"""
import logging

import numpy as np
import media.functions as media
from sklearn.utils import shuffle


PATH_NEGATIVOS = '/home/gmartins/arquivos/uninove/orientacao/projetos/dados/frames/negativos'
PATH_POSITIVOS = '/home/gmartins/arquivos/uninove/orientacao/projetos/dados/frames/positivos'
PATH_WEIGHTS = 'output'
REDUCTION_PRCNT = 87.5
TRAIN_PERCENT = 2/3
logging.basicConfig(format='%(asctime)s - %(name)s: %(message)s', level=logging.INFO)
imgs_negativos = media.read_n_imgs(PATH_NEGATIVOS, REDUCTION_PRCNT)
imgs_positivos = media.read_n_imgs(PATH_POSITIVOS, REDUCTION_PRCNT)
# Mount sets
X = np.concatenate((np.array(imgs_negativos), np.array(imgs_positivos)))
Y = np.concatenate((np.zeros(len(imgs_negativos)), np.ones(len(imgs_positivos))))
# Scale images to the [0,1] range
X = X.astype('float32') / 255
# Clear lists
imgs_negativos = None
imgs_positivos = None
# Define train and test
X, Y = shuffle(X, Y)
train_sample = round(np.size(X, 0) * TRAIN_PERCENT)
X_train = X[:train_sample]
X_test = X[train_sample:]
Y_train = Y[:train_sample]
Y_test = Y[train_sample:]
logging.info('X shape: {}'.format(X.shape))
logging.info('Y shape: {}'.format(Y.shape))
logging.info('X_train shape: {}'.format(X_train.shape))
logging.info('Y_train shape: {}'.format(Y_train.shape))
logging.info('X_test shape: {}'.format(X_test.shape))
logging.info('Y_test shape: {}'.format(Y_test.shape))
# Make sure images have shape (sample_number, 135, 90, 1)
X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)
# Clear sets
X = None
Y = None
# Clear all
X_train = None
Y_train = None
X_test = None
Y_test = None
