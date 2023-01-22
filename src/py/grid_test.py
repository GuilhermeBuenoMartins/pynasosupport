#!/usr/bin/env cola-env
# -*- coding: utf-8 -*-
import os.path
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D

from handlers import log, media, model
from handlers.gridsearch import GridSearch
from morpholayers.layers import Gradient2D, InternalGradient2D


def main():
    datasetPath = input('Type dataset path: ')

    # Settings
    formatTime = '%Y-%m-%d_%H-%M-%S'
    logFile = 'grid_test_{}.log'.format(datetime.now().strftime(formatTime))
    log.set_log(logFile, datasetPath)

    # Constants
    CV = 2
    FACTOR = 0.9
    NUM_CLASSES = 2
    LEARNING_RATE = 0.004

    # Dataset
    negativePath = os.path.join(datasetPath, 'negativos')
    positivePath = os.path.join(datasetPath, 'positivos')
    negativeImgs = media.read_imgs(negativePath, FACTOR)
    positiveImgs = media.read_imgs(positivePath, FACTOR)
    xSet = np.concatenate((np.array(negativeImgs), np.array(positiveImgs)))
    ySet = np.concatenate((np.zeros(len(negativeImgs)), np.ones(len(positiveImgs))))
    negativeImgs = None
    positiveImgs = None
    input_shape = (xSet.shape[1:])

    # Convolution model
    layers = model.getVGG16()
    convModel = model.buidlModel(layers, input_shape)

    # Gradient model
    layers = model.getVGG16()
    layers[0] = Gradient2D(4, (3, 3))
    gradModel = model.buidlModel(layers, input_shape)

    # Internal gradient model
    layers = model.getVGG16()
    layers[0] = InternalGradient2D(4, (3, 3))
    intGradModel = model.buidlModel(layers, input_shape)

    # GridSearch
    models = [convModel, gradModel, intGradModel]
    grid = GridSearch(models, cv=CV)
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    grid.compileModels(optimizer, 'categorical_crossentropy', ['accuracy'])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=.5)]
    grid.fitModels(xSet, ySet, 32, 3, 'auto', callbacks, workers=4, use_multiprocessing=True)

    trainAccuracyMean, testAccuracyMean = grid.getAccuracyMean()
    bestTrainAccuracy, bestTestAccuracy = grid.getBestAccuracy()
    for i in range(len(models)):
        print('\nModel {}:'.format(i))
        print('--------------------------------------------------')
        print('Training: ')
        print('\tAccuracy mean: {}'.format(trainAccuracyMean[i]))
        print('\tBest Accuracy: {}'.format(bestTrainAccuracy[i]))
        print('Test: ')
        print('\tAccuracy mean: {}'.format(testAccuracyMean[i]))
        print('\tBest Accuracy: {}'.format(bestTestAccuracy[i]))


if __name__ == '__main__':
    main()
