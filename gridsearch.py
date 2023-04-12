import numpy as np
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from tensorflow.python.keras.utils.np_utils import to_categorical


class GridSearch:
    cv = None
    model_list = None
    num_classes = None
    acc_mean_list = None

    def __init__(self, model_list: list, num_classes: int = 2, cv: int = 2):
        self.model_list = model_list
        self.num_classes = num_classes
        self.cv = cv

    def mount_sets(self, x, y, train_id, val_id):
        train_x, val_x = x[train_id], x[val_id]
        train_y, val_y = to_categorical(y[train_id], self.num_classes), to_categorical(y[val_id], self.num_classes)
        print('Training sample size: ', train_y.shape[0])
        print('Validation sample size: ', val_y.shape[0])
        return train_x, train_y, val_x, val_y

    def treat_callbacks(self, callbacks_list):
        if callbacks_list is None:
            return [None for i in range(len(self.model_list))]
        return callbacks_list

    def fit_model(self, model, x, y, batch_size=None, epochs=1, verbose='auto', callbacks=None, val_data=None,
                  workers=1, use_multiprocessing=False):
        skf = StratifiedKFold(n_splits=self.cv, shuffle=True)
        acc_list = []
        for fold_id, (train_id, val_id) in enumerate(skf.split(x, y)):
            print('Fold number: ', fold_id + 1)
            train_x, train_y, val_x, val_y = self.mount_sets(x, y, train_id, val_id)
            model.fit(train_x, train_y, batch_size, epochs, verbose, callbacks=callbacks,
                                                validation_data=(val_x, val_y), workers=workers,
                                                use_multiprocessing=use_multiprocessing)
            pred_val_y = np.argmax(model.predict(val_x))
            acc = metrics.accuracy_score(val_y, pred_val_y)
            print('\nFold accuracy: ', acc, '\n')
            acc_list.append(acc)
        print("Model fitted.")
        return np.mean(acc_list)

    def fit_models(self, x, y, batch_size=None, epochs=1, verbose='auto', callbacks_list=None, workers=1,
                   use_multiprocessing=False):
        self.acc_mean_list = []
        callbacks_list = self.treat_callbacks(callbacks_list)
        for model, callbacks in zip(self.model_list, callbacks_list):
            acc_mean = self.fit_model(model, x, y, batch_size, epochs, verbose, callbacks, workers, use_multiprocessing)
            self.acc_mean_list.append(acc_mean)
            print('Accuracy mean: ', acc_mean)
        print('Models fitted!')

    def get_best_model(self):
        best_model_id = np.argmax(self.acc_mean_list)
        print('Best model index: ', best_model_id)
        return self.model_list[best_model_id], best_model_id
