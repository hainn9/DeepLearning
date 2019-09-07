from dl_cv.neuralnetwork import LeNet
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.datasets import cifar10
from keras.optimizers import SGD
from keras import backend as K
import numpy as np
from matplotlib import pyplot as plt

print("[INFO] Get data")
(trainX, trainY), (testX, testY) = cifar10.load_data()
trainX, validX, trainY, validY = train_test_split(trainX, trainY, test_size=0.175)
trainX = trainX.reshape(trainX.shape[0], 32, 32, 3)
validX = validX.reshape(validX.shape[0], 32, 32, 3)
testX = testX.reshape(testX.shape[0], 32, 32, 3)
trainX = trainX.astype("float")/255
validX  =validX.astype("float")/255
testX = testX.astype("float")/255
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
validY = lb.fit_transform(validY)
testY = lb.fit_transform(testY)
label_name = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


print("[INFO] Compile model")
opt = SGD(lr=0.01)
model = LeNet.build(width=32, height=32, depth=3, classes=10)
model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

H = model.fit(trainX, trainY, validation_data=(validX, validY), batch_size=128, epochs=20, verbose=1)

print("[INFO] evaluate model")
pred = model.predict(testX, batch_size=128, verbose=1)
print(classification_report(testY.argmax(axis=1), pred.argmax(axis=1), target_names=label_name))

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
plt.savefig("lenet_cifar10.png")
plt.show()
