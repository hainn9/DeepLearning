import matplotlib
matplotlib.use("Agg")

from dl_cv.neuralnetwork import MiniVGGNet
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD
from keras.datasets import cifar10
from keras.callbacks import ModelCheckpoint
import numpy as np
from matplotlib import pyplot as plt
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weights", required=True, help="Path of weights")
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
sgd = SGD(lr=0.01, decay=0.01/40, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])
## Checkpoint and save whenever validation loss decrease
# fname = os.path.sep.join([args["weights"], "minivgg_cifar10_monitor_weights_{epoch:03d}_{val_loss:.4f}.hdf5"])
# checkpoint = ModelCheckpoint(fname, monitor="val_loss", mode="min", save_best_only=True, verbose=1)

## Checkpoint whenever validation loss decrease and save only the best
checkpoint = ModelCheckpoint(args["weights"], monitor="val_loss", mode="min", save_best_only=True, verbose=1)
callbacks = [checkpoint]
# Train model
print("[INFO] Training model")
model.fit(trainX, trainY, validation_data=(validX, validY), callbacks=callbacks, epochs=40, batch_size=64, verbose=1)
