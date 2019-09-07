from keras.applications import VGG16
from keras.applications import imagenet_utils
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelEncoder
from dl_cv.io import HDF5DatasetWriter
from imutils import paths
import numpy as np 
import os
import argparse
import random
import progressbar 

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Path of input dataset")
ap.add_argument("-o", "--output", required=True, help="Path of output hdf5 file")
ap.add_argument("-b", "--batchSize", default=32, help="Batch size")
ap.add_argument("-s", "--bufferSize", default=1000, help="Size of buffer feature")
args = vars(ap.parse_args())

bs = args["batchSize"]

print("[INFO] Loading dataset")
imagePaths = list(paths.list_images(args["dataset"]))
random.shuffle(imagePaths)

labels = [p.split(os.path.sep)[-2] for p in imagePaths]
le = LabelEncoder()
labels = le.fit_transform(labels)

print("[INFO] Loading model VGG16")
model = VGG16(weights="imagenet", include_top=False)
dataset = HDF5DatasetWriter(dims=(len(imagePaths), 512*7*7), outputPath=args["output"], dataKey="features", buffSize=args["bufferSize"])
dataset.storeClassLabels(le.classes_)

widgets = ["Extracting feature: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(imagePaths), widgets=widgets).start()

for i in np.arange(0, len(imagePaths), bs):
    batchPath = imagePaths[i:i+bs]
    batchLabel = labels[i:i+bs]
    batchImage = []

    for (j, imagepath) in enumerate(batchPath):
        image = load_img(imagepath, target_size=(224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = imagenet_utils.preprocess_input(image)

        batchImage.append(image)
    batchImage = np.vstack(batchImage)
    features = model.predict(batchImage, batch_size=bs)
    features = features.reshape((features.shape[0], 512*7*7))
    dataset.add(features, batchLabel)
    pbar.update(i)

dataset.close()
pbar.finish()


