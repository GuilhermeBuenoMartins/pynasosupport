#!/usr/bin/env cola-env
# -*- coding: utf-8 -*-
"""
Script of layers comparison between morpholayers and convolution layers
"""

import logging as log
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.keras.regularizers import l1_l2
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.regularizers import *
from functions.evaluation import get_model, use_model
from functions.media import read_imgs
from functions.sampler import build_sets
from morpholayers.layers import Dilation2D, Erosion2D, Closing2D, Opening2D, Gradient2D, InternalGradient2D, \
    TopHatOpening2D, TopHatClosing2D
from morpholayers.regularizers import L1L2Lattice

# Constants
FILE_LOG = '../output/logs/file_log.log'
NEGATIVE_PATH = '/home/gmartins/arquivos/uninove/mestrado/orientacao/projetos/dados/frames/negativos'
POSITIVE_PATH = '/home/gmartins/arquivos/uninove/mestrado/orientacao/projetos/dados/frames/positivos'
FACTOR = .9  # Factor of reduction of image
NUM_CLASSES = 2  # Number of classes
NUM_FILTERS = 8  # Number of filters
FILTERS_SIZE = 7  # Square filters size
BATCH_SIZE = 32
EPOCHS = 4
LR = .001  # Learning rate
REGULARIZER_PARAM = .002
TRAIN_PRCNT = 2 / 3  # Percent considered for training data

log.basicConfig(filename=FILE_LOG, level=log.INFO)

negative_imgs = read_imgs(NEGATIVE_PATH, FACTOR)
positive_imgs = read_imgs(POSITIVE_PATH, FACTOR)

x_train, y_train, x_test, y_test = build_sets(negative_imgs, positive_imgs, TRAIN_PRCNT)

negative_imgs, positive_imgs = None, None

y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

input_shape = (x_train.shape[1:])

# Convolution
layer_0 = Conv2D(
    NUM_FILTERS,
    kernel_size=(FILTERS_SIZE, FILTERS_SIZE),
    kernel_regularizer=l1_l2(l1=REGULARIZER_PARAM, l2=REGULARIZER_PARAM),
    activation='relu')
model_conv, _ = get_model(input_shape, NUM_CLASSES, layer_0)
histConv = use_model(x_train, y_train, x_test, y_test, model_conv, BATCH_SIZE, EPOCHS, LR)

# Dilation
layer_0 = Dilation2D(
    NUM_FILTERS,
    kernel_size=(FILTERS_SIZE, FILTERS_SIZE),
    kernel_regularization=L1L2Lattice(l1=REGULARIZER_PARAM, l2=REGULARIZER_PARAM))
model_dil, _ = get_model(input_shape, NUM_CLASSES, layer_0)
histDil = use_model(x_train, y_train, x_test, y_test, model_dil, BATCH_SIZE, EPOCHS, LR)

model_dil = None
# Erosion
layer_0 = Erosion2D(
    NUM_FILTERS,
    kernel_size=(FILTERS_SIZE, FILTERS_SIZE),
    kernel_regularization=L1L2Lattice(l1=REGULARIZER_PARAM, l2=REGULARIZER_PARAM))
model_ero, _ = get_model(input_shape, NUM_CLASSES, layer_0)
histEro = use_model(x_train, y_train, x_test, y_test, model_ero, BATCH_SIZE, EPOCHS, LR)

model_ero = None
# Gradient
layer_0 = Gradient2D(
    NUM_FILTERS,
    kernel_size=(FILTERS_SIZE, FILTERS_SIZE),
    kernel_regularization=L1L2Lattice(l1=REGULARIZER_PARAM, l2=REGULARIZER_PARAM))
model_grad, _ = get_model(input_shape, NUM_CLASSES, layer_0)
histGrad = use_model(x_train, y_train, x_test, y_test, model_grad, BATCH_SIZE, EPOCHS, LR)
model_grad = None

# Internal Gradient
layer_0 = InternalGradient2D(
    NUM_FILTERS,
    kernel_size=(FILTERS_SIZE, FILTERS_SIZE),
    kernel_regularization=L1L2Lattice(l1=REGULARIZER_PARAM, l2=REGULARIZER_PARAM))
model_int_grad, _ = get_model(input_shape, NUM_CLASSES, layer_0)
histInternalGrad = use_model(x_train, y_train, x_test, y_test, model_int_grad, BATCH_SIZE, EPOCHS, LR)
model_int_grad = None

# Openning
layer_0 = Opening2D(
    NUM_FILTERS,
    kernel_size=(FILTERS_SIZE, FILTERS_SIZE),
    kernel_regularization=L1L2Lattice(l1=REGULARIZER_PARAM, l2=REGULARIZER_PARAM))
model_open, _ = get_model(input_shape, NUM_CLASSES, layer_0)
histOpening = use_model(x_train, y_train, x_test, y_test, model_open, BATCH_SIZE, EPOCHS, LR)
model_open = None

# Closing
layer_0 = Closing2D(
    NUM_FILTERS,
    kernel_size=(FILTERS_SIZE, FILTERS_SIZE),
    kernel_regularization=L1L2Lattice(l1=REGULARIZER_PARAM, l2=REGULARIZER_PARAM))
model_clos, _ = get_model(input_shape, NUM_CLASSES, layer_0)
histClosing = use_model(x_train, y_train, x_test, y_test, model_clos, BATCH_SIZE, EPOCHS, LR)
model_clos = None

# White top hat
layer_0 = TopHatOpening2D(
    NUM_FILTERS,
    kernel_size=(FILTERS_SIZE, FILTERS_SIZE),
    kernel_regularization=L1L2Lattice(l1=REGULARIZER_PARAM, l2=REGULARIZER_PARAM))
model_w_tophat, _ = get_model(input_shape, NUM_CLASSES, layer_0)
histTopHatOpening = use_model(x_train, y_train, x_test, y_test, model_w_tophat, BATCH_SIZE, EPOCHS, LR)
model_w_tophat = None

# Black top hat
layer_0 = TopHatClosing2D(
    NUM_FILTERS,
    kernel_size=(FILTERS_SIZE, FILTERS_SIZE),
    kernel_regularization=L1L2Lattice(l1=REGULARIZER_PARAM, l2=REGULARIZER_PARAM))
model_b_tophat, _ = get_model(input_shape, NUM_CLASSES, layer_0)
histTopHatClosing = use_model(x_train, y_train, x_test, y_test, model_b_tophat, BATCH_SIZE, EPOCHS, LR)
model_b_tophat = None

# Results in training accuracy
print('\n\n')
print('BEST RESULTS FOR METHOD IN TRAINING ACCURACY:')
print('Linear Convolution: ', max(histConv.history['accuracy']))
print('Dilation: ', max(histDil.history['accuracy']))
print('Erosion: ', max(histEro.history['accuracy']))
print('Gradient: ', max(histGrad.history['accuracy']))
print('Internal Gradient: ', max(histInternalGrad.history['accuracy']))
print('TopHatOpening: ', max(histTopHatOpening.history['accuracy']))
print('TopHatClosing: ', max(histTopHatClosing.history['accuracy']))

# Results in validation accuracy
print('\n\n')
print('BEST RESULTS FOR METHOD IN VALIDATION ACCURACY:')
print('Linear Convolution: ', max(histConv.history['val_accuracy']))
print('Dilation: ', max(histDil.history['val_accuracy']))
print('Erosin: ', max(histEro.history['val_accuracy']))
print('Gradient: ', max(histGrad.history['val_accuracy']))
print('Internal Gradient: ', max(histInternalGrad.history['val_accuracy']))
print('TopHatOpening: ', max(histTopHatOpening.history['val_accuracy']))
print('TopHatClosing: ', max(histTopHatClosing.history['val_accuracy']))

# Comparison of Validations accuracy
plt.figure(figsize=(20, 20))
plt.plot(histConv.history['val_accuracy'], label='Linear Convolution')
plt.plot(histDil.history['val_accuracy'], label='Dilation')
plt.plot(histEro.history['val_accuracy'], label='Erosion')
plt.plot(histGrad.history['val_accuracy'], label='Gradient')
plt.plot(histInternalGrad.history['val_accuracy'], label='Internal Gradient')
plt.plot(histTopHatOpening.history['val_accuracy'], label='TopHatOpening')
plt.plot(histTopHatClosing.history['val_accuracy'], label='TopHatClosing')
plt.xlabel('Epocs')
plt.ylabel('Validation Accuracy')
plt.grid()
plt.legend()
plt.show()
