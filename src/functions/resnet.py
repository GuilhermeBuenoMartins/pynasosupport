# -*- coding: utf-8 -*-
"""
Functions to implementation of ResNet models
"""

from tensorflow.keras.layers import Activation, Add, BatchNormalization, Conv2D, Dense, Flatten
from tensorflow.keras.layers import GlobalAveragePooling2D, Input, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.models import Model


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


def get_renet34(shape, k=2):
    """
    The function return model ResNet34.

    :param shape: Input shape
    :param k: number of classes
    :return: model ResNet34
    """
    # Input block
    input_model = Input(shape)
    x = ZeroPadding2D((3, 3))(input_model)
    # Convolutional block 1
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
    output_model = Dense(k, activation='softmax')(x)
    return Model(input_model, output_model)
