from keras.layers.core import Dense, Activation, Dropout
from keras.layers.merge import concatenate
from keras.layers.convolutional import Conv2D,MaxPooling2D,AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Input, Flatten
from keras import backend as K

class MiniGoogleNet:
    @staticmethod
    def conv_module(x, K, kX, kY, strides, chanDim, padding="same"):
        x = Conv2D(K, (kX, kY), strides=strides, padding=padding)(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = Activation("relu")(x)
        return x

    @staticmethod
    def inception_module(x, num1x1, num3x3, chanDim):
        con_1x1 = MiniGoogleNet.conv_module(x, num1x1, 1, 1, (1, 1), chanDim)
        con_3x3 = MiniGoogleNet.conv_module(x, num3x3, 3, 3, (1, 1), chanDim)
        x = concatenate([con_1x1, con_3x3], axis=chanDim)
        return x

    @staticmethod
    def downsample_module(x, K, chanDim):
        con_3x3 = MiniGoogleNet.conv_module(x, K, 3, 3, (2, 2), chanDim, padding="valid")
        pool = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
        x = concatenate([con_3x3, pool], axis=chanDim)
        return x

    @staticmethod
    def build(width, height, depth, classes, reg=0.0002):
        inputShape = (width, height, depth)
        chanDim = -1

        if K.image_data_format() == "first_channel":
            inputShape = (depth, width, height)
            chanDim = 1

        inputs = Input(shape=inputShape)
        x = MiniGoogleNet.conv_module(inputs, 96, 3, 3, (1, 1), chanDim)
        x = MiniGoogleNet.inception_module(x, 32, 32, chanDim)
        x = MiniGoogleNet.inception_module(x, 32, 48, chanDim)
        x = MiniGoogleNet.downsample_module(x, 80, chanDim)
        x = MiniGoogleNet.inception_module(x, 112, 48, chanDim)
        x = MiniGoogleNet.inception_module(x, 96, 64, chanDim)
        x = MiniGoogleNet.inception_module(x, 80, 80, chanDim)
        x = MiniGoogleNet.inception_module(x, 48, 96, chanDim)
        x = MiniGoogleNet.downsample_module(x, 96, chanDim)
        x = MiniGoogleNet.inception_module(x, 176, 160, chanDim)
        x = MiniGoogleNet.inception_module(x, 176, 160, chanDim)
        x = AveragePooling2D(pool_size=(7, 7))(x)
        x = Dropout(0.5)(x)

        x = Flatten()(x)
        x = Dense(classes)(x)
        x = Activation("softmax")(x)

        model = Model(inputs, x, name="googlenet")
        return model
