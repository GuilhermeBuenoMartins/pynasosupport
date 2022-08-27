"""
Functions to evaluation implemented convolutional or morphological layers
"""

import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, MaxPooling2D, Conv2D, Flatten, Dropout

def get_model(input_shape, num_classes, layer_0):
    """
    Function to return a simple model net to be comparing.

    :param input_shape: Image input shape.
    :param num_classes: number of classes.
    :param layer_0: type of first layer of model.
    :return: Network model and first layer of model.
    """

    input_x = Input(input_shape)
    layer_x = layer_0(input_x)
    x = MaxPooling2D(pool_size=(2, 2))(layer_x)
    x = Conv2D(32, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    output_x = Dense(num_classes, activation='softmax')(x)
    return Model(input_x, output_x), Model(input_x, layer_x)


def plot_accuracy(hist):
    """
    Function to plot the training and validation accuracy.

    :param hist: data history.
    :return: None.
    """

    plt.figure()
    plt.plot(hist.hist['accuracy'], label='acc')
    plt.plot(hist.hist['val_accuracy'], label='val_acc')
    plt.grid('on')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def plot_loss(hist):
    """
    Function to plot the training and validation loss.

    :param hist: data history.
    :return: None.
    """

    plt.figure()
    plt.plot(hist.hist['loss'], label='loss')
    plt.plot(hist.hist['val_loss'], label='val_loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid('on')
    plt.legend()
    plt.show()


def plot_output_filters(model, x_train):
    """
    Function plot output of filters

    :param model: Trained model.
    :param x_train: set X of training.
    :return: None
    """

    fig = plt.figure()
    Z = model.predict(x_train[0:1, :, :, :])
    for i in range(Z.shape[3]):
        plt.subplot(2, round(Z.shape[3] / 2), round(i + 1))
        plt.imshow(Z[0, :, :, i], cmap='gray', vmax=Z.max(), vmin=Z.min())
        # plt.colorbar()
    fig.suptitle('Output of Learned Filters for an example')
    plt.show()


def plot_filters(model):
    """
    Function plot learned filters

    :param model: Trained model
    :return: None
    """

    Z = model.layers[-1].get_weights()[0]
    fig = plt.figure()
    for i in range(Z.shape[3]):
        plt.subplot(2, round(Z.shape[3] / 2), round(i + 1))
        plt.imshow(Z[:, :, 0, i], cmap='RdBu', vmax=Z.max(), vmin=Z.min())
    fig.suptitle('Learned Filters')
    plt.show()


def plot_confusion_matrix(conf_matrix):
    """
    Function to plot Confusion Matrix

    :param conf_matrix: confusion matrix
    :return: None
    """

    plt.title('Confusion Matrix')
    plt.imshow(conf_matrix, cmap='hot', vmin=0, vmax=1000)
    plt.show()


def generate_confusion_matrix(x_test, y_test, model):
    """
    Function generates a confusion matrix from model utilising test set x e y.

    :param x_test: test set x.
    :param y_test: test set y.
    :param model: trained network model.
    :return:
    """

    y_pred = np.argmax(model.predict(x_test), axis=1)
    conf_matrix = confusion_matrix(y_test, y_pred)
    return conf_matrix


def use_model(x_train, y_train, x_test, y_test, model, batch_size, epochs, lr=.001):
    """
    Function intends to training and validate model using some previous configuration.

    :param model: tensorflow.keras.Model
    :return: data history
    """

    model.summary()
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        metrics=['accuracy'])
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=5,
                restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(
                patience=2,
                factor=.5)],
        verbose=0)
    return history
