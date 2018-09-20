"""MobileNet v2 models for Keras.
# Reference
- [Inverted Residuals and Linear Bottlenecks Mobile Networks for
   Classification, Detection and Segmentation]
   (https://arxiv.org/abs/1801.04381)
"""

from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dropout, Flatten
from keras.layers import Activation, BatchNormalization, add, Reshape, Dense
from keras.applications.mobilenet import relu6, DepthwiseConv2D
from keras.utils.vis_utils import plot_model
from keras.applications import mobilenet

from keras import backend as K


def _conv_block(inputs, filters, kernel, strides):
    """Convolution Block
    This function defines a 2D convolution operation with BN and relu6.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
    # Returns
        Output tensor.
    """

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
    x = BatchNormalization(axis=channel_axis)(x)
    return Activation(relu6)(x)


def _bottleneck(inputs, filters, kernel, t, s, r=False):
    """Bottleneck
    This function defines a basic bottleneck structure.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        r: Boolean, Whether to use the residuals.
    # Returns
        Output tensor.
    """

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    tchannel = K.int_shape(inputs)[channel_axis] * t

    x = _conv_block(inputs, tchannel, (1, 1), (1, 1))

    x = DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation(relu6)(x)

    x = Conv2D(filters, (1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization(axis=channel_axis)(x)

    if r:
        x = add([x, inputs])
    return x


def _inverted_residual_block(inputs, filters, kernel, t, strides, n):
    """Inverted Residual Block
    This function defines a sequence of 1 or more identical layers.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        n: Integer, layer repeat times.
    # Returns
        Output tensor.
    """

    x = _bottleneck(inputs, filters, kernel, t, strides)

    for i in range(1, n):
        x = _bottleneck(x, filters, kernel, t, 1, True)

    return x


def MobileNetv2(input_shape, k):
    """MobileNetv2
    This function defines a MobileNetv2 architectures.
    # Arguments
        input_shape: An integer or tuple/list of 3 integers, shape
            of input tensor.
        k: Integer, number of classes.
    # Returns
        MobileNetv2 model.
    """

    # inputs = Input(shape=input_shape)
    # x = _conv_block(inputs, 32, (3, 3), strides=(2, 2))
    #
    # x = _inverted_residual_block(x, 16, (3, 3), t=1, strides=1, n=1)
    # x = _inverted_residual_block(x, 24, (3, 3), t=6, strides=2, n=2)
    # x = _inverted_residual_block(x, 32, (3, 3), t=6, strides=2, n=3)
    # x = _inverted_residual_block(x, 64, (3, 3), t=6, strides=2, n=4)
    # x = _inverted_residual_block(x, 96, (3, 3), t=6, strides=1, n=3)
    # x = _inverted_residual_block(x, 160, (3, 3), t=6, strides=2, n=3)
    # x = _inverted_residual_block(x, 320, (3, 3), t=6, strides=1, n=1)
    #
    # x = _conv_block(x, 1280, (1, 1), strides=(1, 1))
    # x = GlobalAveragePooling2D()(x)
    # x = Reshape((1, 1, 1280))(x)
    # x = Dropout(0.3, name='Dropout')(x)
    # x = Conv2D(k, (1, 1), padding='same')(x)
    #
    # x = Activation('softmax', name='softmax')(x)
    # output = Reshape((k,))(x)
    #
    # model = Model(inputs, output)
    # plot_model(model, to_file='images/MobileNetv2.png', show_shapes=True)

    # model = Sequential()
    # mob = mobilenet.MobileNet(input_shape=input_shape, alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=True,
    #                           weights='imagenet', input_tensor=None, pooling=None, classes=1000)
    # model.add(Dense(1000, input_shape=(26,)))
    # model.add(mob)
    #
    # model.layers.pop()
    # output = Reshape((26,))
    # # model.add(output)
    # # model2 = Model(model.input, output)
    # for i in model.layers:
    #     print(i)

    # load vgg16 without dense layer and with theano dim ordering
    base_model = mobilenet.MobileNet(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

    # number of classes in your dataset e.g. 20
    num_classes = 26

    x = Flatten()(base_model.output)
    predictions = Dense(num_classes, activation='softmax')(x)

    # create graph of your new model
    head_model = Model(input=base_model.input, output=predictions)

    model = head_model

    return model


if __name__ == '__main__':
    MobileNetv2((224, 224, 3), 100)
