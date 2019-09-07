import matplotlib
matplotlib.use("Agg")

from config import dog_and_cat_config as config
from dl_cv.preprocessor import ImageToArrayPreprocessor
from dl_cv.preprocessor import SimplePreprocessor, MeanPreprocessor, PatchPreprocessor
from dl_cv.callbacks import TrainingMonitor
from dl_cv.neuralnetwork import AlexNet
from dl_cv.io import HDF5DatasetGenerator

from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import os
import json

aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, shear_range=0.2, horizontal_flip=True, fill_mode="nearest")
mean = json.loads(open(config.DATASET_MEAN).read())
sp = SimplePreprocessor(227, 227)
pp = PatchPreprocessor(227, 227)
mp = MeanPreprocessor(mean["R"], mean["G"], mean["B"])
iap = ImageToArrayPreprocessor()

print("[INFO] Preparing data")
trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, 128, [pp, mp, iap], aug, classes=2)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, 128, [sp, mp, iap], classes=2)

print("[INFO] Building model")
opt = Adam(lr=0.001)
model = AlexNet.buidl(width=227, height=227, depth=3, classes=2)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

path = os.path.sep.join([config.OUTPUT_PATH, "_{}.png".format(os.getpid())])
callbacks = [TrainingMonitor(path)]

model.fit_generator(trainGen.generator(), steps_per_epoch=trainGen.numImages // 128,
                    validation_data=valGen.generator(), validation_steps=valGen.numImages // 128,
                    epochs=75, max_queue_size=2*128,
                    callbacks=callbacks, verbose=1)

print("[INFO] Saving model")
model.save(config.MODEL_PATH, overwrite=True)

trainGen.close()
valGen.close()
