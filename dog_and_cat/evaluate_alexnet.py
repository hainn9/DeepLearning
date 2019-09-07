from config import dog_and_cat_config as config
from dl_cv.preprocessor import ImageToArrayPreprocessor,SimplePreprocessor, MeanPreprocessor, CropPreprocessor
from dl_cv.io import HDF5DatasetGenerator
from dl_cv.utils.ranked import rank5_accuracy
from keras.models import load_model
import pickle
import os
import numpy as np
import json
import progressbar

means = json.loads(open(config.DATASET_MEAN).read())

sp = SimplePreprocessor(227, 227)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
cp = CropPreprocessor(227, 227)
iap = ImageToArrayPreprocessor()

print("[INFO] Loading model")
model = load_model(config.MODEL_PATH)

print("[INFO] Predicting on test data no crop")
testGen = HDF5DatasetGenerator(config.TEST_HDF5, 64, [sp, mp, iap], classes=2)
predictions = model.predict_generator(testGen.generator(), steps=testGen.numImages//64, max_queue_size=64*2,verbose=1)

(rank1, _) = rank5_accuracy(predictions, testGen.db["labels"])
print("[INFO] rank-1 accuracy: {:.2f}%".format(rank1 * 100))
testGen.close()

print("[INFO] Predicting on test data with crop-10")
testGen = HDF5DatasetGenerator(config.TEST_HDF5, 64, [mp], classes=2)
predictions = []

widgets = ["Evaluating :", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=testGen.numImages//64, widgets=widgets).start()
for (i, (images, labels)) in enumerate(testGen.generator(passes=1)):
    for image in images:
        crop = cp.preprocess(image)
        crop = np.array([iap.preprocess(c) for c in crop], dtype="float32")
        pred = model.predict(crop)
        predictions.append(pred.mean(axis=0))
    pbar.update(i)
pbar.finish()
(rank1, _) = rank5_accuracy(predictions, testGen.db["labels"])
print("[INFO] rank-1 accuracy: {:.2f}%".format(rank1 * 100))
testGen.close()
