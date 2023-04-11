import numpy as np
from sklearn.model_selection import StratifiedKFold
from tensorflow.python.keras.utils.np_utils import to_categorical


class GridSearch:
    cv = None
    model_list = None
    num_classes = None
    acc_list = None
    loss_list = None

    def __init__(self, model_list: list, num_classes: int = 2, cv: int = 2):
        self.model_list = model_list
        self.num_classes = num_classes
        self.cv = cv

    def mount_sets(self, x, y, train_id, val_id):
        train_x, val_x = x[train_id], x[val_id]
        if self.num_classes > 1:
            train_y, val_y = to_categorical(y[train_id], self.num_classes), to_categorical(y[val_id], self.num_classes)
        else:
            train_y, val_y = y[train_id], y[val_id]
        print('Training sample size: ', train_y.shape[0])
        print('Validation sample size: ', val_y.shape[0])
        return train_x, train_y, (val_x, val_y)

    def fit_model(self, fold_id, train_x, train_y, batch_size=None, epochs=1, verbose='auto', callbacks_list=None,
                  val_data=None, workers=1, use_multiprocessing=False):
        if callbacks_list is None:
            hist = self.model_list[fold_id].fit(train_x, train_y, batch_size, epochs, verbose, callbacks=None,
                                                validation_data=val_data, workers=workers,
                                                use_multiprocessing=use_multiprocessing)
        else:
            hist = self.model_list[fold_id].fit(train_x, train_y, batch_size, epochs, verbose,
                                                callbacks=callbacks_list[fold_id], validation_data=val_data,
                                                workers=workers, use_multiprocessing=use_multiprocessing)
        return hist

    def get_best_model(self):
        last_acc_list = [np.max(acc['val']) for acc in self.acc_list]
        best_model_id = np.argmax(last_acc_list)
        print('The best model index: ', best_model_id)
        return self.model_list[best_model_id]

    def fit_models(self, x, y, batch_size=None, epochs=1, verbose='auto', callbacks_list=None, workers=1,
                   use_multiprocessing=False):
        skf = StratifiedKFold(n_splits=self.cv, shuffle=True)
        self.acc_list = []
        self.loss_list = []
        for fold_id, (train_id, val_id) in enumerate(skf.split(x, y)):
            print('Fold number: ', fold_id + 1)
            train_x, train_y, val_data = self.mount_sets(x, y, train_id, val_id)
            hist = self.fit_model(fold_id, train_x, train_y, batch_size, epochs, verbose, callbacks_list, val_data,
                                  workers, use_multiprocessing)
            self.acc_list.append({'train': hist.history['accuracy'], 'val': hist.history['val_accuracy']})
            self.loss_list.append({'train': hist.history['loss'], 'val': hist.history['val_loss']})
        print('Models fitted')
