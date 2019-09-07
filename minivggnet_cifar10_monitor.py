import matplotlib
matplotlib.use("Agg")

from dl_cv.neuralnetwork import MiniVGGNet
from dl_cv.callbacks import TrainingMonitor
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD
from keras.datasets import cifar10
from keras.callbacks import LearningRateScheduler
import numpy as np 
from matplotlib import pyplot as plt 
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="Path of output")
args = vars(ap.parse_args())

print("[INFO] Process ID {}".format(os.getpid()))

# Load dataset
print("[INFO] Loading dataset")
(trainX, trainY),(testX, testY) = cifar10.load_data()
trainX, validX, trainY, validY = train_test_split(trainX, trainY, test_size=0.175)
trainX = trainX.reshape(trainX.shape[0], 32, 32, 3)
testX = testX.reshape(testX.shape[0], 32, 32, 3)
validX = validX.reshape(validX.shape[0], 32, 32, 3)
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0
validX = validX.astype("float") / 255.0

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)
validY = lb.fit_transform(validY)

label_name = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# Build model
sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])
figPath = os.path.sep.join([args["output"], "minivgg_cifar10_monitor_{}.png".format(os.getpid())])
jsonPath = os.path.sep.join([args["output"], "minivgg_cifar10_monitor_{}.json".format(os.getpid())])
callbacks = [TrainingMonitor(figPath, jsonPath)]
# Train model
print("[INFO] Training model")
model.fit(trainX, trainY, validation_data=(validX, validY), callbacks=callbacks, epochs=100, batch_size=64, verbose=1)
