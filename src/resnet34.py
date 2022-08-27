#!/usr/bin/env cola-env
# -*- coding: utf-8 -*-
"""
Script of ResNet34 execution
"""
import logging
from src.functions import resnet
import numpy as np
import tensorflow as tf
import src.media.functions as media
from sklearn.utils import shuffle

PATH_NEGATIVOS = '/home/gmartins/mestrado/orientacao/projetos/dados/frames/negativos'
PATH_POSITIVOS = '/home/gmartins/mestrado/orientacao/projetos/dados/frames/positivos'
PATH_WEIGHTS = 'output'
REDUCTION_PRCNT = 75.0
TRAIN_PERCENT = 200/3   # 2/3
logging.basicConfig(format='%(asctime)s - %(name)s: %(message)s', level=logging.INFO)
imgs_negativos = media.read_n_imgs(PATH_NEGATIVOS, REDUCTION_PRCNT)
imgs_positivos = media.read_n_imgs(PATH_POSITIVOS, REDUCTION_PRCNT)
# Mount sets
X = np.concatenate((np.array(imgs_negativos), np.array(imgs_positivos)))
Y = np.concatenate((np.zeros(len(imgs_negativos)), np.ones(len(imgs_positivos))))
# Clear lists
imgs_negativos = None
imgs_positivos = None
# Define train and test
X, Y = shuffle(X, Y)
train_sample = round(np.size(X, 0) * TRAIN_PERCENT)
X_train = X[:train_sample]
Y_train = Y[:train_sample]
X_test = X[train_sample:]
Y_test = Y[train_sample:]
logging.info('X_train shape: {}'.format(X_train.shape))
logging.info('Y_train shape: {}'.format(Y_train.shape))
logging.info('X_test shape: {}'.format(X_test.shape))
logging.info('Y_test shape: {}'.format(Y_test.shape))
# Clear sets
X = None
Y = None
# Train model
resnet34 = functions.get_renet34(X_train.shape[1:])
resnet34.summary()
resnet34.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
# TODO: epochs=100
# TODO: Validation
resnet34.fit(X_train, Y_train, epochs=16, batch_size=32)
# TODO: Test
# TODO: Plotagem
resnet34.save_weights(PATH_WEIGHTS)

