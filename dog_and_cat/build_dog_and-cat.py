from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from config import dog_and_cat_config as config
from dl_cv.io import HDF5DatasetWriter
from dl_cv.preprocessor import AspectAwarePreprocessor

import progressbar
import numpy as np
import json
from imutils import paths
import os
import cv2

print("[INFO] Loading and Preprocessing data")
trainDatas = list(paths.list_images(config.IMAGES_PATH))
trainLabels = [p.split(os.path.sep)[2].split(".")[0] for p in trainDatas]

le = LabelEncoder()
trainLabels = le.fit_transform(trainLabels)
split = train_test_split(trainDatas, trainLabels, test_size=config.NUM_TEST_IMAGES, random_state=42, stratify=trainLabels)
(trainDatas, testDatas, trainLabels, testLabels) = split
split = train_test_split(trainDatas, trainLabels, test_size=config.NUM_VAL_IMAGES, random_state=42, stratify=trainLabels)
(trainDatas, valDatas, trainLabels, valLabels) = split

datasets = [
    ("train", trainDatas, trainLabels, config.TRAIN_HDF5),
    ("test", testDatas, testLabels, config.TEST_HDF5),
    ("val", valDatas, valLabels, config.VAL_HDF5)
]

aap = AspectAwarePreprocessor(256, 256)
(R, G, B) = ([], [], [])

for (dType, datas, labels, outputPath) in datasets:
    print("[INFO] Building {}".format(outputPath))
    writer = HDF5DatasetWriter((len(datas), 256, 256, 3), outputPath)

    widgets = ["Building datasets: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(datas), widgets=widgets).start()

    for(i, (data, label)) in enumerate(zip(datas, labels)):
        image = cv2.imread(data)
        image = aap.preprocess(image)

        if dType == "train":
            (b, g, r) = cv2.mean(image)[:3]
            R.append(r)
            G.append(g)
            B.append(b)
        writer.add([image], [label])
        pbar.update(i)

    pbar.finish()
    writer.close()

print("[INFO] Serializing mean")
D = {"R": np.mean(R), "G": np.mean(G), "B": np.mean(B)}
f = open(config.DATASET_MEAN, "w")
f.write(json.dumps(D))
f.close()
