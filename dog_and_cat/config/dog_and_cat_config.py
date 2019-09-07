# path of image directory
IMAGES_PATH = "../datasets/kaggle_dog_and_cat/train"

NUM_CLASSES = 2
NUM_IMAGE = 25000
NUM_VAL_IMAGES = 1250*NUM_CLASSES
NUM_TEST_IMAGES = 1250*NUM_CLASSES

TRAIN_HDF5 = "../datasets/kaggle_dog_and_cat/hdf5/train.hdf5"
VAL_HDF5 = "../datasets/kaggle_dog_and_cat/hdf5/val.hdf5"
TEST_HDF5 = "../datasets/kaggle_dog_and_cat/hdf5/test.hdf5"

# path of model
MODEL_PATH = "../output/kaggle_dog_and_cat/alexnet_dog_and_cat.model"

DATASET_MEAN = "../output/kaggle_dog_and_cat/dog_and_cat_mean.json"

EXTRACTED_FEATURE_PATH = "../output/kaggle_dog_and_cat/resnet_feature.hdf5"

OUTPUT_PATH = "../output"
