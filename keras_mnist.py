from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras import backend as K
import numpy as np 
from matplotlib import pyplot as plt 
import argparse
#%matplotlib inline

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="Path for report plot")
args = vars(ap.parse_args())

# Load dataset
print("[INFO] Loading MNIST dataset")
X, Y = datasets.fetch_openml('mnist_784', version=1, return_X_y=True)
# X = X.reshape((X.shape[0], -1))
(trainX, testX, trainY, testY) = train_test_split(X, Y, test_size=0.25)

# Encoding label
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

# Build model
model = Sequential()
model.add(Dense(units=256, input_shape=(784,), activation="sigmoid"))
model.add(Dense(units=128, activation="sigmoid"))
model.add(Dense(units=10, activation="softmax"))

# Train model
print("[INFO] Training model")
sgd = SGD(0.01)
model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])
H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=100, batch_size=128)

# Evaluate model
print("[INFO] Evaluating model")
predictions = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), 
                            target_names=[str(x) for x in lb.classes_]))

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
plt.savefig(args["output"])

