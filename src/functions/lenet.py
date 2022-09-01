"""
Functions to implementation of LeNet models
"""
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense


def get_lenet5(input_shape, num_classes, layer_0=None):
    """
    The function to return a LeNet-5 model and the model first layer.

    :param input_shape: image input shape.
    :param num_classes: number of classes.
    :param layer_0: type of first layer of model.
    :return: LetNet-5 model and the models with the first layer.
    """

    input_x = Input(input_shape)
    if layer_0 is None:
        layer_0 = Conv2D(filters=6, kernel_size=(3, 3), activation='relu')
    layer_x = layer_0(input_x)
    x = AveragePooling2D()(layer_x)
    x = Flatten()(x)
    x = Dense(units=120, activation='relu')(x)
    x = Dense(units=84, activation='relu')(x)
    output_x = Dense(units=num_classes, activation='softmax')(x)
    return Model(input_x, output_x), Model(input_x, layer_x)
