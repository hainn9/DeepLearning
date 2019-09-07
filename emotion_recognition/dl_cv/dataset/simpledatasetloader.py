import numpy as np
import cv2
import os

class SimpleDatasetLoader:
    def __init__(self, preprocessors):
        self.preprocessors = preprocessors
        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, imagePaths, verbose = -1):
        data = []
        labels = []

        for (i, imagepath) in enumerate(imagePaths):
            image = cv2.imread(imagepath)
            label = imagepath.split(os.path.sep)[-2]
            if self.preprocessors is not None:
                for p in self.preprocessors:
                    image = p.preprocess(image)

            data.append(image)
            labels.append(label)

            if((verbose > 0) and (i > 0) and ((i+1)%verbose==0)):
                print("[INFO] Processed {}/{}".format(i+1, len(imagePaths)))

        return (np.array(data), np.array(labels))
