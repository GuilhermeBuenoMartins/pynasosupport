#!/usr/bin/env cola-env
# -*- coding: utf-8 -*-
import logging
import os.path
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from handlers import log, media, model, evaluation
from handlers.gridsearch import GridSearch
from sklearn import metrics

from tensorflow.keras.layers import Conv2D

def main():
    datasetPath = input('Type dataset path: ')

    # Settings
    formatTime = '%Y-%m-%d_%H-%M-%S'
    logFile = 'grid_test_{}.log'.format(datetime.now().strftime(formatTime))
    log.set_log(logFile, datasetPath)

    # Constants
    CV = 10
    TRAIN_VALIDATION_FOLDER = 'treinamento_validacao'
    TEST_FOLDER = 'teste'
    OUTPUT_SIZE = (72, 108)

    # Dataset
    negativePath = os.path.join(datasetPath, TRAIN_VALIDATION_FOLDER, 'negativos')
    positivePath = os.path.join(datasetPath, TRAIN_VALIDATION_FOLDER, 'positivos')
    negativeImgs = media.read_images(negativePath, OUTPUT_SIZE)
    positiveImgs = media.read_images(positivePath, OUTPUT_SIZE)
    xSet = np.concatenate((np.array(negativeImgs), np.array(positiveImgs)))
    ySet = np.concatenate((np.zeros(len(negativeImgs)), np.ones(len(positiveImgs))))
    negativeImgs = None
    positiveImgs = None

    # Cross validation
    inputShape = xSet.shape[1:]
    layers = model.getSimpleModel()
    layers[0] = Conv2D(12, (3, 3), activation='relu'),
    convModel = model.buidlModel(layers, inputShape)
    models = [convModel]
    grid = GridSearch(models, cv=CV)
    optimizer = tf.keras.optimizers.Adam()
    grid.compileModels(optimizer, 'categorical_crossentropy', ['accuracy'])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=.5)]
    grid.fitModels(xSet, ySet, epochs=100, verbose='auto', callbacks=callbacks, workers=4, use_multiprocessing=True)

    # Plots
    # Accuracy
    plt.figure()
    plt.title('Convolution accuracy using 4 filters 3 X 3')
    plt.plot(grid.train[0]['accuracy'], label='Training accuracy')
    plt.plot(grid.val[0]['accuracy'], label='Validation accuracy')
    plt.grid('on')
    plt.xlim([0, 100])
    plt.ylim([0.9, 1])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    #
    plt.figure()
    plt.title('Convolution loss using 4 filters 3 X 3')
    plt.plot(grid.train[0]['loss'], label='Training loss')
    plt.plot(grid.val[0]['loss'], label='Validation loss')
    plt.grid('on')
    plt.xlim([0, 100])
    plt.ylim([0, 1])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Test
    negativePath = os.path.join(datasetPath, TEST_FOLDER, 'negativos')
    positivePath = os.path.join(datasetPath, TEST_FOLDER, 'positivos')
    negativeImgs = media.read_images(negativePath, OUTPUT_SIZE)
    positiveImgs = media.read_images(positivePath, OUTPUT_SIZE)
    testX = np.concatenate((np.array(negativeImgs), np.array(positiveImgs)))
    testY = np.concatenate((np.zeros(len(negativeImgs)), np.ones(len(positiveImgs))))
    negativeImgs = None
    positiveImgs = None

    # Confusion matrix
    confusionMatrix = metrics.confusion_matrix(testY, np.argmax(models[0].predict(testX), axis=1))
    confusionMatrixDisplay = metrics.ConfusionMatrixDisplay(confusionMatrix, display_labels=['Negative', 'Positive'])
    confusionMatrixDisplay = confusionMatrixDisplay.plot()
    confusionMatrixDisplay.ax_.set_title('Confusion matrix of convolution using 4 filters 3 x 3')
    plt.show()
    

if __name__ == '__main__':
    main()
