from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.datasets import cifar10
from keras.optimizers import SGD
import numpy as np 
from matplotlib import pyplot as plt 
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="Output path of accuracy result")
agrs = vars(ap.parse_args())

# Load dataset
print("[INFO] Load cifar10 dataset")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

trainX = trainX.reshape((trainX.shape[0], 3072))
testX = testX.reshape((testX.shape[0], 3072))

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

label_name = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# Build model
model = Sequential()
model.add(Dense(units=1024, input_shape=(3072,), activation="relu"))
model.add(Dense(units=512, activation="relu"))
model.add(Dense(units=10, activation="softmax"))

# Train model
print("[INFO] Training model")
sgd = SGD(0.01)
model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])
H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=100, batch_size=32)

# Evaluate model
print("[INFO] Evaluating model")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=label_name))

# Save output to file
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
plt.savefig(agrs["output"])




