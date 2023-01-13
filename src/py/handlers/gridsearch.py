import logging

import numpy as np
from sklearn.model_selection import StratifiedKFold
from tensorflow.python.keras.utils.np_utils import to_categorical


class GridSearch:
    cv = None
    models = None
    num_classes = None
    trainAccuracies = None
    testAccuracies = None

    def __init__(self, model, num_classes: int = 2, cv: int = 2):
        self.models = model
        self.num_classes = num_classes
        self.cv = cv

    def compileModels(self, optimizer='rmsprop', loss=None, metrics=None):

        for modelId in range(len(self.models)):
            model = self.models[modelId]
            model.compile(optimizer, loss, metrics)
            logging.info('Model %i compiled', modelId)

    def fitModels(self, x, y, batch_size=None, epochs=1, verbose='auto', callbacks=None, validation_split=0.0,
                  validation_data=None, workers=1, use_multiprocessing=False):

        modelsQtd = len(self.models)
        self.trainAccuracies = np.zeros((modelsQtd, self.cv))
        self.testAccuracies = np.zeros((modelsQtd, self.cv))
        skf = StratifiedKFold(n_splits=self.cv, shuffle=True)
        for modelId in range(modelsQtd):
            logging.info('Model: %i', modelId)
            for fold, (trainId, testId) in enumerate(skf.split(x, y)):
                logging.info('Fold: %i', fold)
                trainX, trainY = x[trainId], to_categorical(y[trainId], self.num_classes-1)
                testX, testY = x[testId], to_categorical(y[testId], self.num_classes-1)
                logging.info('Train sample size: %s', trainY.shape[0])
                logging.info('Test sample size: %s', testY.shape[0])
                self.models[modelId].fit(trainX, trainY, batch_size, epochs, verbose, callbacks,
                                         validation_split, validation_data, workers=workers,
                                         use_multiprocessing=use_multiprocessing)
                predictedTrainY = self.models[modelId].predict(trainX)
                predictedTestY = self.models[modelId].predict(testX)
                self.trainAccuracies[modelId, fold] = np.mean(predictedTrainY == trainY)
                self.testAccuracies[modelId, fold] = np.mean(predictedTestY == testY)
                logging.info('Train accuracy: %f', self.trainAccuracies[modelId, fold])
                logging.info('Test accuracy: %f', self.testAccuracies[modelId, fold])

