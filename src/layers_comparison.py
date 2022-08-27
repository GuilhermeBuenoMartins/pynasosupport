#!/usr/bin/env cola-env
# -*- coding: utf-8 -*-
"""
Script of layers comparison between morpholayers and conv-layers
"""
# ------------------------
# Dataset extraction
# ------------------------
# Imports
import logging

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
import media.functions as media

# Constants
NUM_CLASSES = 2
PATH_NEGATIVOS = '/home/gmartins/arquivos/uninove/mestrado/orientacao/projetos/dados/frames/negativos'
PATH_POSITIVOS = '/home/gmartins/arquivos/uninove/mestrado/orientacao/projetos/dados/frames/positivos'
REDUCTION_PRCNT = 87.5
TRAIN_PERCENT = 2 / 3
# Extraction
logging.basicConfig(format='%(asctime)s - %(name)s: %(message)s', level=logging.INFO)
imgs_negativos = media.read_n_imgs(PATH_NEGATIVOS, REDUCTION_PRCNT)
imgs_positivos = media.read_n_imgs(PATH_POSITIVOS, REDUCTION_PRCNT)
# Mount sets
X = np.concatenate((np.array(imgs_negativos), np.array(imgs_positivos)))
Y = np.concatenate((np.zeros(len(imgs_negativos)), np.ones(len(imgs_positivos))))
# Scale images to the [0,1] range
X = X.astype('float32') / 255
# Clear lists
imgs_negativos = None
imgs_positivos = None
# Define train and test
X, Y = shuffle(X, Y)
train_sample = round(np.size(X, 0) * TRAIN_PERCENT)
X_train = X[:train_sample]
X_test = X[train_sample:]
Y_train = Y[:train_sample]
Y_test = Y[train_sample:]
logging.info('X shape: {}'.format(X.shape))
logging.info('Y shape: {}'.format(Y.shape))
logging.info('X_train shape: {}'.format(X_train.shape))
logging.info('Y_train shape: {}'.format(Y_train.shape))
logging.info('X_test shape: {}'.format(X_test.shape))
logging.info('Y_test shape: {}'.format(Y_test.shape))
# Clear sets
X = None
Y = None
# Convert class vectors to binary class matrices
Y_train = tf.keras.utils.to_categorical(Y_train, NUM_CLASSES)
Y_test = tf.keras.utils.to_categorical(Y_test, NUM_CLASSES)

# ------------------------
# Models building
# ------------------------
# Imports
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
# # Tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dropout, Dense
# Morpholayers
from src.morpholayers import ToggleMapping2D
from src.morpholayers.regularizers import L1L2Lattice

# Contants
INPUT_SHAPE = (135, 90, 3)
EPOCHS = 2
BATCH_SIZE = 32
NUM_FILTERS = 8
FILTERS_SIZE = 7
REGULARIZER_PARAM = .002


# Functions
def get_model(layer0):
    xin = Input(shape=INPUT_SHAPE)
    xlayer = layer0(xin)
    x = MaxPooling2D(pool_size=(2, 2))(xlayer)
    x = Conv2D(32, kernel_size=(3, 3), activation="relu")(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    xoutput = Dense(NUM_CLASSES, activation="softmax")(x)
    return Model(xin, outputs=xoutput), Model(xin, outputs=xlayer)


def plot_history(history):
    plt.figure()
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid('on')
    plt.legend()
    plt.show()
    plt.plot(history.history['accuracy'], label='acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.grid('on')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def plot_output_filters(model):
    fig = plt.figure()
    Z = model.predict(X_train[0:1, :, :, :])
    for i in range(Z.shape[3]):
        plt.subplot(2, round(Z.shape[3] / 2), round(i + 1))
        plt.imshow(Z[0, :, :, i], cmap='gray', vmax=Z.max(), vmin=Z.min())
        # plt.colorbar()
    fig.suptitle('Output of Learned Filters for an example')
    plt.show()


def plot_filters(model):
    Z = model.layers[-1].get_weights()[0]
    fig = plt.figure()
    for i in range(Z.shape[3]):
        plt.subplot(2, round(Z.shape[3] / 2), round(i + 1))
        plt.imshow(Z[:, :, 0, i], cmap='RdBu', vmax=Z.max(), vmin=Z.min())
    fig.suptitle('Learned Filters')
    plt.show()


def see_results_layer(layer, lr=.001):
    modeli, modellayer = get_model(layer)
    modeli.summary()
    modeli.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                   metrics=["accuracy"])
    historyi = modeli.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_test, Y_test),
                          callbacks=[
                              tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True),
                              tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=.5)], verbose=0)
    y_test = np.argmax(Y_test, axis=1)  # Convert one-hot to index
    y_pred = np.argmax(modeli.predict(X_test), axis=1)
    CM = confusion_matrix(y_test, y_pred)
    print(CM)
    plt.imshow(CM, cmap='hot', vmin=0, vmax=1000)
    plt.title('Confusion Matrix')
    plt.show()
    print(classification_report(y_test, y_pred))
    plot_history(historyi)
    plot_filters(modellayer)
    plot_output_filters(modellayer)
    return historyi


# Convolution
# histConv = see_results_layer(Conv2D(
#     NUM_FILTERS, kernel_size=(FILTERS_SIZE, FILTERS_SIZE),
#     kernel_regularizer=tf.keras.regularizers.l1_l2(l1=REGULARIZER_PARAM, l2=REGULARIZER_PARAM),
#     activation="relu"), lr=.01)
# # Dilation
# histDil = see_results_layer(Dilation2D(
#     NUM_FILTERS, kernel_size=(FILTERS_SIZE, FILTERS_SIZE),
#     kernel_regularization=L1L2Lattice(l1=REGULARIZER_PARAM, l2=REGULARIZER_PARAM)), lr=.01)
# # Erosion
# histEro = see_results_layer(Erosion2D(
#     NUM_FILTERS, padding='valid', kernel_size=(FILTERS_SIZE, FILTERS_SIZE),
#     kernel_regularization=L1L2Lattice(l1=REGULARIZER_PARAM, l2=REGULARIZER_PARAM)), lr=.01)
# # Gradient
# histGrad = see_results_layer(Gradient2D(
#     NUM_FILTERS, padding='valid', kernel_size=(FILTERS_SIZE, FILTERS_SIZE),
#     kernel_regularization=L1L2Lattice(l1=REGULARIZER_PARAM, l2=REGULARIZER_PARAM)), lr=.01)
# # Internal Gradient
# histInternalGrad = see_results_layer(InternalGradient2D(
#     NUM_FILTERS, padding='same', kernel_size=(FILTERS_SIZE, FILTERS_SIZE),
#     kernel_regularization=L1L2Lattice(l1=REGULARIZER_PARAM, l2=REGULARIZER_PARAM)), lr=.01)
# Toggle mapping
# FIXME
histToggle = see_results_layer(ToggleMapping2D(
    num_filters=NUM_FILTERS, kernel_size=(FILTERS_SIZE, FILTERS_SIZE),
    kernel_regularization=L1L2Lattice(l1=REGULARIZER_PARAM, l2=REGULARIZER_PARAM)), lr=.01)
# Opening
# histOpening = see_results_layer(Opening2D(
#     NUM_FILTERS, padding='valid', kernel_size=(FILTERS_SIZE, FILTERS_SIZE),
#     kernel_regularization=L1L2Lattice(l1=REGULARIZER_PARAM, l2=REGULARIZER_PARAM)), lr=.01)
# # Closing
# histClosing = see_results_layer(Closing2D(
#     NUM_FILTERS, padding='valid', kernel_size=(FILTERS_SIZE, FILTERS_SIZE),
#     kernel_regularization=L1L2Lattice(l1=REGULARIZER_PARAM, l2=REGULARIZER_PARAM)), lr=.01)
# # White top hat
# histTopHatOpening = see_results_layer(TopHatOpening2D(
#     NUM_FILTERS, kernel_size=(FILTERS_SIZE, FILTERS_SIZE),
#     kernel_regularization=L1L2Lattice(l1=REGULARIZER_PARAM, l2=REGULARIZER_PARAM)), lr=.01)
# # Black top hat
# histTopHatClosing = see_results_layer(TopHatClosing2D(
#     NUM_FILTERS, kernel_size=(FILTERS_SIZE, FILTERS_SIZE),
#     kernel_regularization=L1L2Lattice(l1=REGULARIZER_PARAM, l2=REGULARIZER_PARAM)), lr=.01)

# ------------------------
# Summary of results
# ------------------------
# Results in training accuracy
print('\n\n')
print('BEST RESULTS FOR METHOD IN TRAINING ACCURACY:')
# print('Linear Convolution: ', max(histConv.history['accuracy']))
# print('Dilation: ', max(histDil.history['accuracy']))
# print('Erosion: ', max(histEro.history['accuracy']))
# print('Gradient: ', max(histGrad.history['accuracy']))
# print('Internal Gradient: ', max(histInternalGrad.history['accuracy']))
print('Toggle: ', max(histToggle.history['accuracy']))
# print('TopHatOpening: ', max(histTopHatOpening.history['accuracy']))
# print('TopHatClosing: ', max(histTopHatClosing.history['accuracy']))
# Results in validation accuracy
print('\n\n')
print('BEST RESULTS FOR METHOD IN VALIDATION ACCURACY:')
# print('Linear Convolution: ', max(histConv.history['val_accuracy']))
# print('Dilation: ', max(histDil.history['val_accuracy']))
# print('Erosin: ', max(histEro.history['val_accuracy']))
# print('Gradient: ', max(histGrad.history['val_accuracy']))
# print('Internal Gradient: ', max(histInternalGrad.history['val_accuracy']))
# print('Toggle: ', max(histToggle.history['val_accuracy']))
# print('TopHatOpening: ', max(histTopHatOpening.history['val_accuracy']))
# print('TopHatClosing: ', max(histTopHatClosing.history['val_accuracy']))
# Comparison of Validations accuracy
plt.figure(figsize=(20, 20))
# plt.plot(histConv.history['val_accuracy'], label='Linear Convolution')
# plt.plot(histDil.history['val_accuracy'], label='Dilation')
# plt.plot(histEro.history['val_accuracy'], label='Erosion')
# plt.plot(histGrad.history['val_accuracy'], label='Gradient')
# plt.plot(histInternalGrad.history['val_accuracy'], label='Internal Gradient')
plt.plot(histToggle.history['val_accuracy'], label='Toggle')
# plt.plot(histTopHatOpening.history['val_accuracy'], label='TopHatOpening')
# plt.plot(histTopHatClosing.history['val_accuracy'], label='TopHatClosing')
plt.xlabel('Epocs')
plt.ylabel('Validation Accuracy')
plt.grid()
plt.legend()
plt.show()
