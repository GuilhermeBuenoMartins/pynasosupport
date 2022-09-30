"""
Functions of support for known models.
"""
import logging

from tensorflow.keras import Sequential, Input
from tensorflow.keras.callbacks import History
from tensorflow.keras.layers import AveragePooling2D, Conv2D, Dense, Flatten, Layer

from morpholayers.layers import Closing2D, Dilation2D, Erosion2D, Gradient2D, InternalGradient2D, Opening2D, \
    TopHatClosing2D, TopHatOpening2D


def get_layer(layer_type: str, num_filters: int, kernel_size: tuple, strides=(1, 1), padding='same',
              activation=None) -> Layer:
    """
    Function to get specific layer with determined parameters.

    :param layer_type: Type of layer:
        'Conv2D';
         'Dilation2D';
          'Erosion2D';
           'Gradient2D';
            'InternalGradient2D';
             'Closing2D';
              'Opening2D';
               'TopHatClosing2D';or
                'TopHatOpening2D'.
    :param params: Layer parameters: filter, kernel_size, stride, padding and activation.
    :return: Layer
    """
    if layer_type is "Conv2D":
        return Conv2D(num_filters, kernel_size, strides, padding, activation=activation)
    elif layer_type is "Dilation2D":
        return Dilation2D(num_filters, kernel_size, strides, padding, activation=activation)
    elif layer_type is "Erosion2D":
        return Erosion2D(num_filters, kernel_size, strides, padding, activation=activation)
    elif layer_type is "Gradient2D":
        return Gradient2D(num_filters, kernel_size, strides, padding)
    elif layer_type is "InternalGradient2D":
        return InternalGradient2D(num_filters, kernel_size, strides, padding)
    elif layer_type is "Opening2D":
        return Opening2D(num_filters, kernel_size, strides, padding)
    elif layer_type is "Closing2D":
        return Closing2D(num_filters, kernel_size, strides, padding)
    elif layer_type is "TopHatOpening2D":
        return TopHatOpening2D(num_filters, kernel_size, strides, padding)
    elif layer_type is "TopHatClosing2D":
        return TopHatClosing2D(num_filters, kernel_size, strides, padding)
    else:
        logging.critical('Layer type {type} not found.'.format(layer_type))
    return None


def get_lenet5(input_shape: tuple, num_classes: int, layer_type: str, num_filters: int, kernel_size: tuple,
               strides=(1, 1), padding='same', activation=None) -> Sequential:
    """
    The function returns LeNet-5 model compiled.

    :param input_shape: Input shape network.
    :param num_classes: Number of classes.
    :param layer_type: Type of first layer:
        'Conv2D';
         'Dilation2D';
          'Erosion2D';
           'Gradient2D';
            'InternalGradient2D';
             'Closing2D';
              'Opening2D';
               'TopHatClosing2D';or
                'TopHatOpening2D'.
    :param layer_params: Layer parameters: filter, kernel_size, stride, padding and activation.
    :return: Sequential model.
    """
    model = Sequential()
    model.add(Input(input_shape))
    model.add(get_layer(layer_type, num_filters, kernel_size, strides, padding, activation))
    model.add(AveragePooling2D())
    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    return model

def fit_model(model, data_train:tuple, batch_size:int, epochs:int, verbose, validation_data:tuple) -> History:
    """
    The function uses paramter to repass to fit model.

    :param model: The model
    :param data_train: X and Y training
    :param batch_size: Number of batch size
    :param epochs: Number of epochs
    :param validation_data: The validation data
    :return: History
    """
    return model.fit(data_train[0], data_train[1], batch_size, epochs, verbose, validation_data=validation_data)
