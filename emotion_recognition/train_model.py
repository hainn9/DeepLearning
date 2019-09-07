import matplotlib
matplotlib.use("Agg")

from config import emotion_recognition_config as config
from dl_cv.callbacks import TrainingMonitor
from dl_cv.callbacks import EpochCheckpoint
from dl_cv.preprocesor import ImageToArrayPreprocessor
from dl_cv.neuralnetwork.emotion_nn import EmotionVGG
from dl_cv.io import HDF5DatasetGenerator

from keras.optimizers import Adam
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
import argparse
import os


ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoint", required=True, help="Path of output checkpoint directory")
ap.add_argument("-m", "--model", type=str, help="Path of model")
ap.add_argument("-s", "--startEpoch", default=0, type=int, help="restart train model at epoch")
ap.add_argument("-l", "--lr", type=float, default=1e-3, help="learning rate value")
args = vars(ap.parse_args())

trainAug = ImageDataGenerator(rotation_range=10, zoom_range=0.1, horizontal_flip=True, rescale=1/255.0, fill_mode="nearest")
valAug = ImageDataGenerator(rescale=1/255.0)
iap = ImageToArrayPreprocessor()

trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, config.BATCH_SIZE, [iap], trainAug, classes=config.NUM_CLASSES)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, config.BATCH_SIZE, [iap], valAug, classes=config.NUM_CLASSES)

if args["model"] is None:
    print("[INFO] Compiling model")
    model = EmotionVGG.build(width=48, height=48, depth=1, classes=config.NUM_CLASSES)
    opt = Adam(lr=args["lr"])
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
else:
    print("[INFO] Loading model {}".format(args["model"]))
    model = load_model(args["model"])

    print("[INFO] old learning rate: {}".format(K.get_value(model.optimizer.lr)))
    K.set_value(model.optimizer.lr, args["lr"])
    print("[INFO] new learning rate: {}".format(K.get_value(model.optimizer.lr)))

figPaths = os.path.sep.join([config.OUTPUT_PATH, "vggnet_emotion.png"])
jsonPaths = os.path.sep.join([config.OUTPUT_PATH, "vggnet_emotion.json"])
callbacks = [
    EpochCheckpoint(args["checkpoint"], every=5, startAt=args["startEpoch"]),
    TrainingMonitor(figPaths, jsonPath=jsonPaths, startAt=args["startEpoch"])
]

model.fit_generator(trainGen.generator(), steps_per_epoch=trainGen.numImages//config.BATCH_SIZE,
                    validation_data=valGen.generator(), validation_steps=valGen.numImages//config.BATCH_SIZE,
                    max_queue_size=2*config.BATCH_SIZE, epochs=75,
                    callbacks=callbacks, verbose=1)

trainGen.close()
valGen.close()
