from keras.models import Sequential
from keras.layers.core import Dense,Activation,Dropout,Flatten
from keras.regularizers import l2
from keras import backend as K
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.layers.normalization import BatchNormalization

class AlexNet:
    @staticmethod
    def buidl(width, height, depth, classes, reg=0.0002):
        inputShape = (width, height, depth)
        chanDim = -1

        if K.image_data_format() == "first_channel":
            inputShape = (depth, width, height)
            chanDim = 1

        model = Sequential()
        model.add(Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4),padding="same", kernel_regularizer=l2(reg), input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(filters=256, kernel_size=(5, 5), padding="same", kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(filters=384, kernel_size=(3, 3), padding="same", kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(filters=384, kernel_size=(3, 3), padding="same", kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(filters=384, kernel_size=(3, 3), padding="same", kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(4096, kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Dropout(0.5))

        model.add(Dense(4096, kernel_regularizer=l2(reg)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Dropout(0.5))

        model.add(Dense(classes, kernel_regularizer=l2(reg)))
        model.add(Activation("softmax"))

        return model
