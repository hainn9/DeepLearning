from dl_cv.preprocesor import AspectAwarePreprocessor
from dl_cv.preprocesor import ImageToArrayPreprocessor
from dl_cv.dataset import SimpleDatasetLoader
from dl_cv.neuralnetwork import MiniVGGNet

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from imutils import paths
import imutils
from matplotlib import pyplot as plt 
import numpy as np 
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Path to dataset")
args = vars(ap.parse_args())

print("[INFO] Preprocessing data")
imagePaths = list(paths.list_images(args["dataset"]))
classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]
classNames = [str(x) for x in np.unique(classNames)]

aap = AspectAwarePreprocessor(64, 64)
iap = ImageToArrayPreprocessor()

sdl = SimpleDatasetLoader(preprocessors=[aap, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.astype("float") / 255.0

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

# Build model
print("[INFO] Building model")
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, zoom_range=0.2, fill_mode="nearest", horizontal_flip=True)
sgd = SGD(lr=0.05)
model = MiniVGGNet.build(width=64, height=64, depth=3, classes=len(classNames))
model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])

print("[INFO] Training model")
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=32), validation_data=(testX, testY), steps_per_epoch=len(trainX), epochs=100, verbose=1)

# Evaluate model
print("[INFO] Evaluating model")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=classNames))

# Plot learning curve
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_acc"], label="val_acc")
plt.title("Training Lost and Accuracy")
plt.xlabel("#Epochs")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig("minivggnet_flower_augmentation.png")
plt.show()
