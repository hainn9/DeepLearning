from dl_cv.neuralnetwork import LeNet

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import img_to_array
from keras.utils import np_utils
from matplotlib import pyplot as plt 
import numpy as np 
import os
import imutils
from imutils import paths
import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Path of dataset")
ap.add_argument("-m", "--model", required=True, help="Path of output model")
args = vars(ap.parse_args())

data = []
labels = []
print("[INFO] Preparing data")
for imagePath in sorted(list(paths.list_images(args["dataset"]))):
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = imutils.resize(image, width=28)
    image = img_to_array(image)
    data.append(image)

    label = imagePath.split(os.path.sep)[-3]
    label = "smiling" if label == "positives" else "not_simling"
    labels.append(label)

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

le = LabelEncoder().fit(labels)
labels = np_utils.to_categorical(le.transform(labels), 2)
print("[INFO] Finish prepare data")
classTotal = labels.sum(axis=0)
classWeight = classTotal.max() / classTotal

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=42)

# Build model
model = LeNet.build(width=28, height=28, depth=1, classes=2)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train model
print("[INFO] Training model")
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=64, epochs=20, verbose=1, class_weight=classWeight)

# Evaluate model
predictions = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=le.classes_))

print("[INFO] Saving model")
model.save(args["model"])

# plot learning curve
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 20), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 20), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 20), H.history["acc"], label="train_accuracy")
plt.plot(np.arange(0, 20), H.history["val_acc"], label="val_accuracy")
plt.title("Loss and Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig("lenet_smiling_detection.png")
plt.show()
