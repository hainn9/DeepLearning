import matplotlib
matplotlib.use("Agg")

from dl_cv.neuralnetwork import MiniVGGNet
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD
from keras.datasets import cifar10
from keras.callbacks import LearningRateScheduler
import numpy as np 
from matplotlib import pyplot as plt 
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="Path of output")
ap.add_argument("-t", "--type", default="Standard", help="Type of learning rate schedule")
args = vars(ap.parse_args())

def step_decay(epoch):
    initAlpha = 0.01
    factor = 0.25
    dropEvery = 5

    alpha = initAlpha * (factor ** np.floor((1 + epoch)/dropEvery))

    return float(alpha)

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
## improve accuracy and avoid overfitting technique
## by using learning rate schedule: 
## Standard -> change(decrease) learning rate every epochs
if args["type"] == "Standard":
    sgd = SGD(lr=0.01, decay=0.01/40, momentum=0.9, nesterov=True)
    callbacks = []

## Step-based -> change learning rate in specific epochs
if args["type"] == "Step-based":
    sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
    callbacks = [LearningRateScheduler(step_decay)]
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])

# Train model
print("[INFO] Training model")
H = model.fit(trainX, trainY, validation_data=(validX, validY), callbacks=callbacks, epochs=40, batch_size=64, verbose=1)

# Evaluate model
print("[INFO] Evaluating model")
predictions = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=label_name))

# Plot accuracy/loss
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 40), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 40), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 40), H.history["acc"], label="train_accuracy")
plt.plot(np.arange(0, 40), H.history["val_acc"], label="val_accuracy")
plt.title("Loss and Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig("minivggnet_cifar10.png")
plt.show()