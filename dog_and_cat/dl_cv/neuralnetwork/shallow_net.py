from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Flatten,Activation,Dense
from keras import backend as K

class ShallowNet:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (width, height, depth)

        if K.image_data_format() == "channels_first":
            inputShape = (depth, width, height)

        model.add(Conv2D(32, (3,3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model
