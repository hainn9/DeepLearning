from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import  Conv2D, MaxPooling2D
from keras.layers.normalization import  BatchNormalization
from keras.models import  Sequential
from keras.layers.advanced_activations import  ELU
from keras import backend as K

class EmotionVGG:
    @staticmethod
    def build(width, height, depth, classes):
        inputShape = (width, height, depth)
        chanDim = -1

        if K.image_data_format() == "channel_first":
            inputShape = (depth, width, height)
            chanDim = 1

        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding="same", kernel_initializer="he_normal", input_shape=inputShape))
        model.add((ELU()))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(32, (3, 3), kernel_initializer="he_normal", padding="same"))
        model.add(ELU())
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding="same", kernel_initializer="he_normal", input_shape=inputShape))
        model.add((ELU()))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), kernel_initializer="he_normal", padding="same"))
        model.add(ELU())
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, (3, 3), padding="same", kernel_initializer="he_normal", input_shape=inputShape))
        model.add((ELU()))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), kernel_initializer="he_normal", padding="same"))
        model.add(ELU())
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(64, kernel_initializer="he_normal"))
        model.add(ELU())
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(64, kernel_initializer="he_normal"))
        model.add(ELU())
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(classes, kernel_initializer="he_normal"))
        model.add(Activation("softmax"))

        return model
