import matplotlib
matplotlib.use("Agg")

from dl_cv.neuralnetwork import MiniGoogleNet
from dl_cv.callbacks import TrainingMonitor
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.datasets import cifar10
from keras.callbacks import LearningRateScheduler
import numpy as np
from matplotlib import pyplot as plt
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="Path of output model")
ap.add_argument("-o", "--output", required=True, help="Path of output")
ap.add_argument("-t", "--type", default="Step-based", help="Type of learning rate schedule")
args = vars(ap.parse_args())

NUM_EPOCHS = 70
INIT_LR = 5e-3

def poly_decay(epoch):
    maxEpochs = NUM_EPOCHS
    baseLR = INIT_LR
    power = 1.0

    alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power

    return float(alpha)

# Load dataset
print("[INFO] Loading dataset")
(trainX, trainY),(testX, testY) = cifar10.load_data()
trainX = trainX.astype("float")
testX = testX.astype("float")

mean = np.mean(trainX, axis=0)
trainX -= mean
testX -= mean

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

label_name = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

aug = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True, fill_mode="nearest")

figPaths = os.path.sep.join([args["output"], "_{}.png".format(os.getpid())])
jsonPaths = os.path.sep.join([args["output"], "_{}.json".format(os.getpid())])

# Build model
## improve accuracy and avoid overfitting technique
## by using learning rate schedule:
## Standard -> change(decrease) learning rate every epochs
if args["type"] == "Standard":
    sgd = SGD(lr=0.01, decay=0.01/40, momentum=0.9, nesterov=True)
    callbacks = []

## Step-based -> change learning rate in specific epochs
if args["type"] == "Step-based":
    sgd = SGD(lr=INIT_LR, momentum=0.9, nesterov=True)
    callbacks = [TrainingMonitor(figPaths, jsonPath=jsonPaths), LearningRateScheduler(poly_decay)]
model = MiniGoogleNet.build(width=32, height=32, depth=3, classes=10)
model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])

# Train model
print("[INFO] Training model")
model.fit_generator(aug.flow(trainX, trainY, batch_size=64), validation_data=(testX, testY), steps_per_epoch=len(trainX)//64, epochs=NUM_EPOCHS, callbacks=callbacks, verbose=1)

# Save model
print("[INFO] Saving model")
model.save(args["model"])
