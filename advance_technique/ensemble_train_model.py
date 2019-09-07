import matplotlib
matplotlib.use("Agg")

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.datasets import cifar10
from keras.optimizers import SGD
from dl_cv.neuralnetwork import MiniVGGNet
from keras.preprocessing.image import ImageDataGenerator

import numpy as np
from matplotlib import pyplot as plt
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="Path of output model")
ap.add_argument("-o", "--output", required=True, help="Path of output")
ap.add_argument("-n", "--num_models", type=int, default=5, help="Number of model to train")
args = vars(ap.parse_args())

((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship" "truck"]
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearst")

for i in np.arange(args["num_models"]):
    print("[INFO] Training model {}/{}".format(i+1, args["num_models"]))
    opt = SGD(lr=0.01, decay=0.01/40, momentum=0.9, nesterov=True)
    model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit_generator(aug.flow(trainX, trainY, batch_size=64), validation_data=(testX, testY), epochs=40, steps_per_epoch=len(trainX)//64, verbose=1)
    p = [args["model"], "model_{}.model".format(i)]
    model.save(os.path.sep.join(p))

    predictions = model.predict(testX, batch_size=64)
    report = classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames)
    p = [args["model"], "model_{}.png".format(i)]
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0,40), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0,40), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0,40), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0,40), H.history["val_acc"], label="val_acc")
    plt.title("Learning curve for model {}".format(i))
    plt.xlabel("Epochs")
    plt.ylabel("Acc/Loss")
    plt.legend()
    plt.savefig(os.path.sep.join(p))
    plt.close()
