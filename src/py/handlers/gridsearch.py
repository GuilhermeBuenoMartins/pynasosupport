import logging

import numpy as np
from sklearn.model_selection import StratifiedKFold
from tensorflow.python.keras.utils.np_utils import to_categorical


class GridSearch:
    cv = None
    models = None
    num_classes = None
    train = None
    val = None

    def __init__(self, models: list, num_classes: int = 2, cv: int = 2):
        """ This class groups models to apply grid search.

        Returns:
            object: A GridSearch object with some attributes as: models, num_classes, cv,
                trainAccuracies and testAccuracies. The attribute 'models' is a list of
                models for application of grid search, the num_classes values is the number
                of classication classes, 'cv' defines cross val, 'trainAccuracies'
                is a matrix with models' training accuracies values while 'testAccuracies',
                models' test accuracies values.

        """
        self.models = models
        self.num_classes = num_classes
        self.cv = cv

    @DeprecationWarning
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

    def fitModels(self, x, y, batch_size=None, epochs=1, verbose='auto', callbacks=None, workers=1,
                  use_multiprocessing=False):
        """Applies training using cross val defined in class instantiation.

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
        skf = StratifiedKFold(n_splits=self.cv, shuffle=True)
        self.train = []
        self.val = []
        for foldId, (trainId, valId) in enumerate(skf.split(x, y)):
            logging.info('Fold id %i', foldId)
            if (self.num_classes > 1) :
                trainX, trainY = x[trainId], to_categorical(y[trainId], self.num_classes)
                valX, valY = x[valId], to_categorical(y[valId], self.num_classes)
            else:
                trainX, trainY = x[trainId], y[trainId]
                valX, valY = x[valId], y[valId]
            logging.info('Train sample size: %s', trainY.shape[0])
            logging.info('Validation sample size: %s', valY.shape[0])
            hist = None
            if callbacks is None:
                hist = self.models[foldId].fit(trainX, trainY, batch_size, epochs, verbose, callbacks,
                                            validation_data=(valX, valY), workers=workers,
                                            use_multiprocessing=use_multiprocessing)
            else:
                hist = self.models[foldId].fit(trainX, trainY, batch_size, epochs, verbose, callbacks,
                                               validation_data=(valX, valY), workers=workers,
                                               use_multiprocessing=use_multiprocessing)
            self.train.append(
                {'accuracy': np.mean(hist.history['accuracy']), 'loss': np.mean(hist.history['loss'], 0)})
            self.val.append(
                {'accuracy': np.mean(hist.history['val_accuracy']), 'loss': np.mean(hist.history['val_loss'], 0)})
        logging.info('Models fitted.')
        return self.models[np.argmax(np.mean(val['accuracy']) for val in self.val)]
