from dl_cv.preprocesor import ImageToArrayPreprocessor
from dl_cv.preprocesor import AspectAwarePreprocessor
from dl_cv.dataset import SimpleDatasetLoader
from dl_cv.neuralnetwork import FCHeadNet

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from keras.applications import VGG16
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.layers import Input
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator

from imutils import paths
import os
import argparse
import numpy as np 
from matplotlib import pyplot as plt 

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Path of input dataset")
ap.add_argument("-m", "--model", required=True, help="Path of output model")
args = vars(ap.parse_args())

print("[INFO] Loading dataset")
imagePaths = list(paths.list_images(args["dataset"]))
classNames = [pt.split(os.path.sep)[-2] for pt in imagePaths]
classNames = [str(x) for x in np.unique(classNames)]

aap = AspectAwarePreprocessor(224, 224)
iap = ImageToArrayPreprocessor()

sdl = SimpleDatasetLoader(preprocessors=[aap, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
print("[INFO] Loading dataset finish")

print("[INFO] Preprocessing data")
data = data.astype("float")/255.0

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, zoom_range=0.2, 
                            fill_mode="nearest", horizontal_flip=True)

print("[INFO] Buiding base model")
rms = RMSprop(lr=0.001)
baseModel = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
headModel = FCHeadNet.build(baseModel, len(classNames), 256)
model = Model(inputs=baseModel.input, outputs=headModel)

# Freeze all base model
for layer in baseModel.layers:
    layer.trainable = False

print("[INFO] Training model in warm up phase")
model.compile(optimizer=rms, loss="categorical_crossentropy", metrics=["accuracy"])
# For warm up phase, use 10-30 epoch to train model
model.fit_generator(aug.flow(trainX, trainY, batch_size=32), validation_data=(testX, testY), epochs=25, steps_per_epoch=len(trainX) // 32, verbose=1)

print("[INFO] Evaluating model after warm up phase")
preds = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), preds.argmax(axis=1), target_names=classNames))

print("[INFO] Fine-tuning phase")
for layer in baseModel.layers[15:]:
    layer.trainalbe = True

print("[INFO] Training model in fine tune phase")
sgd = SGD(lr=0.001)
model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])
model.fit_generator(aug.flow(trainX, trainY, batch_size=32), validation_data=(testX, testY), epochs=100, steps_per_epoch=len(trainX) // 32, verbose=1)

print("[INFO] Evaluating model in fine tune phase")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=classNames))

print("[INFO] Saving model")
model.save(args["model"])

