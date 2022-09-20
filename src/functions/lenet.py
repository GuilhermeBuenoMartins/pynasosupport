"""
Functions to implementation of LeNet models
"""

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense
from tensorflow.python.keras import Input, Model


def get_lenet5(input_shape, num_classes, optimizer='Adam', layer_0=None):
    """
    The function to return a LeNet-5 model and the model first layer.

    :param input_shape: image input shape.
    :param num_classes: number of classes.
    :param layer_0: type of first layer of model.
    :return: LetNet-5 model and the models with the first layer.
    """

    model = Sequential()
    if layer_0 is None:
        layer_0 = Conv2D(filters=6, kernel_size=(3, 3), input_shape=input_shape, activation='relu')
    model.add(layer_0)
    model.add(AveragePooling2D())
    model.add(Flatten())
    model.add(Dense(units=120, activation='relu'))
    model.add(Dense(units=84, activation='relu'))
    model.add(Dense(units=num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model
