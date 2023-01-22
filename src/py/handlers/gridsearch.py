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

    def __init__(self, models: list, num_classes: int = 2, cv: int = 2):
        """ This class groups models to apply grid search.

        Returns:
            object: A GridSearch object with some attributes as: models, num_classes, cv,
                trainAccuracies and testAccuracies. The attribute 'models' is a list of
                models for application of grid search, the num_classes values is the number
                of classication classes, 'cv' defines cross validation, 'trainAccuracies'
                is a matrix with models' training accuracies values while 'testAccuracies',
                models' test accuracies values.

        """
        self.models = models
        self.num_classes = num_classes
        self.cv = cv

    def compileModels(self, optimizer='rmsprop', loss=None, metrics=None):
        """Configures the models in list for training.

        Args:
            optimizer: String (name of optimizer) or optimizer instance.
                              See `tf.keras.optimizers`.
            loss: String (name of objective function), objective function or
                              `tf.keras.losses.Loss` instance. See `tf.keras.losses`. An objective
                              function is any callable with the signature
                              `scalar_loss = fn(y_true, y_pred)`. If the model has multiple
                              outputs, you can use a different loss on each output by passing a
                              dictionary or a list of losses. The loss value that will be
                              minimized by the model will then be the sum of all individual
                              losses.
            metrics: List of metrics to be evaluated by the model during training
                                 and testing. Typically you will use `metrics=['accuracy']`.
                                 To specify different metrics for different outputs of a
                                 multi-output model, you could also pass a dictionary, such as
                                 `metrics={'output_a': 'accuracy', 'output_b': ['accuracy', 'mse']}`.
                                 You can also pass a list (len = len(outputs)) of lists of metrics
                                 such as `metrics=[['accuracy'], ['accuracy', 'mse']]` or
                                 `metrics=['accuracy', ['accuracy', 'mse']]`.
        """
        for modelId in range(len(self.models)):
            model = self.models[modelId]
            model.compile(optimizer, loss, metrics)
            logging.info('Model %i compiled', modelId)

    def fitModels(self, x, y, batch_size=None, epochs=1, verbose='auto', callbacks=None, validation_split=0.0,
                  validation_data=None, workers=1, use_multiprocessing=False):
        """Applies training using cross validation defined in class instantiation.

        Args:
            x: Input data. It could be:
              - A Numpy array (or array-like), or a list of arrays
                (in case the model has multiple inputs).
              - A TensorFlow tensor, or a list of tensors
                (in case the model has multiple inputs).
              - A dict mapping input names to the corresponding array/tensors,
                if the model has named inputs.
              - A `tf.data` dataset. Should return a tuple
                of either `(inputs, targets)` or
                `(inputs, targets, sample_weights)`.
              - A generator or `keras.utils.Sequence` returning `(inputs, targets)`
                or `(inputs, targets, sample weights)`.
            y: Target data. Like the input data `x`,
              it could be either Numpy array(s) or TensorFlow tensor(s).
              It should be consistent with `x` (you cannot have Numpy inputs and
              tensor targets, or inversely). If `x` is a dataset, generator,
              or `keras.utils.Sequence` instance, `y` should
              not be specified (since targets will be obtained from `x`).
            batch_size: Integer or `None`.
                Number of samples per gradient update.
                If unspecified, `batch_size` will default to 32.
                Do not specify the `batch_size` if your data is in the
                form of symbolic tensors, datasets,
                generators, or `keras.utils.Sequence` instances (since they generate
                batches).
            epochs: Integer. Number of epochs to train the model.
                An epoch is an iteration over the entire `x` and `y`
                data provided.
                Note that in conjunction with `initial_epoch`,
                `epochs` is to be understood as "final epoch".
                The model is not trained for a number of iterations
                given by `epochs`, but merely until the epoch
                of index `epochs` is reached.
            verbose: 0, 1, or 2. Verbosity mode.
                0 = silent, 1 = progress bar, 2 = one line per epoch.
                Note that the progress bar is not particularly useful when
                logged to a file, so verbose=2 is recommended when not running
                interactively (eg, in a production environment).
            callbacks: List of `keras.callbacks.Callback` instances.
                List of callbacks to apply during training.
                See `tf.keras.callbacks`.
            validation_split: Float between 0 and 1.
                Fraction of the training data to be used as validation data.
                The model will set apart this fraction of the training data,
                will not train on it, and will evaluate
                the loss and any model metrics
                on this data at the end of each epoch.
                The validation data is selected from the last samples
                in the `x` and `y` data provided, before shuffling. This argument is
                not supported when `x` is a dataset, generator or
               `keras.utils.Sequence` instance.
            validation_data: Data on which to evaluate
                the loss and any model metrics at the end of each epoch.
                The model will not be trained on this data.
                `validation_data` will override `validation_split`.
                `validation_data` could be:
                  - tuple `(x_val, y_val)` of Numpy arrays or tensors
                  - tuple `(x_val, y_val, val_sample_weights)` of Numpy arrays
                  - dataset
                For the first two cases, `batch_size` must be provided.
                For the last case, `validation_steps` could be provided.
            workers: Integer. Used for generator or `keras.utils.Sequence` input
                only. Maximum number of processes to spin up
                when using process-based threading. If unspecified, `workers`
                will default to 1. If 0, will execute the generator on the main
                thread.
            use_multiprocessing: Boolean. Used for generator or
                `keras.utils.Sequence` input only. If `True`, use process-based
                threading. If unspecified, `use_multiprocessing` will default to
                `False`. Note that because this implementation relies on
                multiprocessing, you should not pass non-picklable arguments to
                the generator as they can't be passed easily to children processes.
            """
        modelsQtd = len(self.models)
        self.trainAccuracies = np.zeros((modelsQtd, self.cv))
        self.testAccuracies = np.zeros((modelsQtd, self.cv))
        skf = StratifiedKFold(n_splits=self.cv, shuffle=True)
        for modelId in range(modelsQtd):
            logging.info('Model: %i', modelId)
            for fold, (trainId, testId) in enumerate(skf.split(x, y)):
                logging.info('Fold: %i', fold)               
                trainX, trainY = x[trainId], to_categorical(y[trainId], self.num_classes)
                testX, testY = x[testId], to_categorical(y[testId], self.num_classes)
                logging.info('Train sample size: %s', trainY.shape[0])
                logging.info('Test sample size: %s', testY.shape[0])
                self.models[modelId].fit(trainX, trainY, batch_size, epochs, verbose, callbacks,
                                         validation_split, validation_data, workers=workers,
                                         use_multiprocessing=use_multiprocessing)
                predictedTrainY = self.models[modelId].predict(trainX)
                predictedTestY = self.models[modelId].predict(testX)
                self.trainAccuracies[modelId, fold] = np.mean(np.argmax(predictedTrainY, 1) == np.argmax(trainY, 1))
                self.testAccuracies[modelId, fold] = np.mean(np.argmax(predictedTestY, 1) == np.argmax(testY, 1))
                logging.info('Train accuracy: %f', self.trainAccuracies[modelId, fold])
                logging.info('Test accuracy: %f', self.testAccuracies[modelId, fold])
        logging.info('Models fitted.')

    def getAccuracyMean(self, accuracy: str = 'auto'):
        """Returns test or train accuracy mean.

        Args:
            accuracy: String parameter accepts one of following values: auto, test or train.

        Returns:
            For accuracy equals to 'auto', it will be returned two numpy array, training and test accuracy mean
            respectively. Otherwise, it will be return only test accuracy mean, weather accuracy equals to 'test',
            or train accuracy mean, when accuracy equals to 'train'.
        """
        if self.trainAccuracies is None or self.testAccuracies is None:
            logging.warnning('Models not fitted. Use GridSearch().fit(x, y) to get using this function.')
        else:
            if accuracy is 'auto':
                return np.mean(self.trainAccuracies, 1), np.mean(self.testAccuracies, 1)
            if accuracy is 'test':
                return np.mean(self.testAccuracies, 1)
            if accuracy is 'train':
                return np.mean(self.trainAccuracies, 1)
            else:
                logging.warnning('Parameter "accuracy" should be auto, test or train.')

    def getBestAccuracy(self, accuracy: str = 'auto'):
        """Returns the best train or test accuracies for each model.

        Args:
            accuracy: String parameter accepts one of following values: auto, test or train.

        Returns:
              For accuracy equals to 'auto', it will be returned two numpy array, best training and test
              accuracy respectively. Otherwise, it will be return only best test accuracy, weather accuracy
              equals to 'test', or best train accuracy, when accuracy equals to 'train'.
        """
        if self.trainAccuracies is None or self.testAccuracies is None:
            logging.warnning('Models not fitted. Use GridSearch().fit(x, y) to get using this function.')
        else:
            if accuracy is 'auto':
                bestTrainAccuracy = np.max(self.trainAccuracies, 1)
                bestTestAccuracy = np.max(self.testAccuracies, 1)
                return bestTrainAccuracy, bestTestAccuracy
            if accuracy is 'test':
                bestAccuracy = np.max(self.testAccuracies, 1)
            if accuracy is 'train':
                bestAccuracy = np.max(self.trainAccuracies, 1)
                return bestAccuracy
            else:
                logging.warnning('Parameter "accuracy" should be auto, test or train.')
            return bestAccuracy
