"""
Functions of support for known models.
"""
import logging

import tensorflow.python.keras
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras.callbacks import History
from tensorflow.keras.layers import Activation, AveragePooling2D, Add, BatchNormalization, Conv2D, Dense, Dropout, \
    Flatten, GlobalAveragePooling2D, Layer, MaxPool2D, MaxPooling2D, ZeroPadding2D

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


def identity_block(x, filters_num):
    """
    Identity block without downsample

    :param x: input
    :param filters_num: number of filters
    :return: output
    """

    x_skip = x
    # Layer 1
    x = Conv2D(filters_num, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # Layer 2
    x = Conv2D(filters_num, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    # Add resisdue
    x = Add()([x_skip, x])
    x = Activation('relu')(x)
    return x


def conv_block(x, filters_num):
    """
    Convolutional block with downsample

    :param x: input
    :param filters_num: number of filters
    :return: output
    """

    x_skip = x
    # Layer 1
    x = Conv2D(filters_num, (3, 3), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # Layer 2
    x = Conv2D(filters_num, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    # Convolution of residue
    x_skip = Conv2D(filters_num, (1, 1), strides=(2, 2), padding='same')(x_skip)
    # Add residue
    x = Add()([x_skip, x])
    x = Activation('relu')(x)
    return x


def get_func_layer(x, layer_type: str, num_filters: int, kernel_size: tuple, strides=(1, 1), padding='same',
                   activation=None):
    """
    Function to get specific layer with determined parameters.

    :param x: input x
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
        return Conv2D(num_filters, kernel_size, strides, padding, activation=activation)(x)
    elif layer_type is "Dilation2D":
        return Dilation2D(num_filters, kernel_size, strides, padding, activation=activation)(x)
    elif layer_type is "Erosion2D":
        return Erosion2D(num_filters, kernel_size, strides, padding, activation=activation)(x)
    elif layer_type is "Gradient2D":
        return Gradient2D(num_filters, kernel_size, strides, padding)(x)
    elif layer_type is "InternalGradient2D":
        return InternalGradient2D(num_filters, kernel_size, strides, padding)(x)
    elif layer_type is "Opening2D":
        return Opening2D(num_filters, kernel_size, strides, padding)(x)
    elif layer_type is "Closing2D":
        return Closing2D(num_filters, kernel_size, strides, padding)(x)
    elif layer_type is "TopHatOpening2D":
        return TopHatOpening2D(num_filters, kernel_size, strides, padding)(x)
    elif layer_type is "TopHatClosing2D":
        return TopHatClosing2D(num_filters, kernel_size, strides, padding)(x)
    else:
        logging.critical('Layer type {type} not found.'.format(layer_type))
    return None


def get_resnet34(input_shape: tuple, num_classes: int, layer_type: str, num_filters: int, kernel_size: tuple,
                 strides=(1, 1), padding='same', activation=None) -> Model:
    """
    The function return model ResNet34.

    :param shape: Input shape
    :param k: number of classes
    :return: model ResNet34
    """
    # Input block
    input_model = Input(input_shape)
    x = ZeroPadding2D((3, 3))(input_model)
    # Block 1
    x = get_func_layer(x, layer_type, num_filters, kernel_size, strides, padding, activation)
    x = Conv2D(64, (7, 7), strides=(2, 2))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    # Convolutional block 2
    x = identity_block(x, 64)
    x = identity_block(x, 64)
    x = identity_block(x, 64)
    # Convolutional block 3
    x = conv_block(x, 128)
    x = identity_block(x, 128)
    x = identity_block(x, 128)
    x = identity_block(x, 128)
    # Convolutional block 4
    x = conv_block(x, 256)
    x = identity_block(x, 256)
    x = identity_block(x, 256)
    x = identity_block(x, 256)
    x = identity_block(x, 256)
    x = identity_block(x, 256)
    # Convolutional block 5
    x = conv_block(x, 512)
    x = identity_block(x, 512)
    x = identity_block(x, 512)
    # Output block
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    output_model = Dense(num_classes, activation='softmax')(x)
    model = Model(input_model, output_model)
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    return model


def buidlModel(layers: list = [], input_shape=None) -> Model:
    numLayers = len(layers)
    logging.info('Model with {} layer will be built.'.format(numLayers))
    if numLayers is 0:
        logging.critical('Parameter can not be "None". Build a model through a layers list.')
        return None
    inputLayer = Input(input_shape)
    x = layers[0](inputLayer)
    for layerId in range(1, numLayers - 1):
        x = layers[layerId](x)
    outputLayer = layers[-1](x)
    model = Model(inputLayer, outputLayer)
    logging.info('Model created successfully.')
    logging.info(model.summary())
    return model


def getSimpleModel(num_classes: int = 2) -> list:
    layers = [
        Conv2D(4, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(32, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ]
    logging.info('Layers created.')
    return layers


def getAlexNet(num_classes: int = 2) -> list:
    layers = [
        Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation='relu'),
        BatchNormalization(),
        MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
        Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding="same"),
        BatchNormalization(),
        MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
        Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
        BatchNormalization(),
        Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
        BatchNormalization(),
        Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
        BatchNormalization(),
        MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ]
    logging.info('AlexNet layers created.')
    return layers


def getVGG16(num_classes: int = 2) -> list:
    layers = [
        # Fist Convolutional Block
        Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
        Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
        MaxPool2D(pool_size=2, strides=2, padding='same'),
        # Second Convolutional Block
        Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'),
        Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'),
        MaxPool2D(pool_size=2, strides=2, padding='same'),
        # Third Convolutional Block
        Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'),
        Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'),
        MaxPool2D(pool_size=2, strides=2, padding='same'),
        # Fourth Convolutional Block
        Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'),
        Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'),
        MaxPool2D(pool_size=2, strides=2, padding='same'),
        # Fiveth Convolutional Block
        Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'),
        Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'),
        Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'),
        MaxPool2D(pool_size=2, strides=2, padding='same'),
        # Dense layers
        Flatten(),
        Dense(units=4096, activation='relu'),
        Dense(units=4096, activation='relu'),
        Dense(units=1000, activation='relu'),
        Dense(num_classes, activation='softmax')
    ]
    logging.info('VGG-16 layers created.')
    return layers
