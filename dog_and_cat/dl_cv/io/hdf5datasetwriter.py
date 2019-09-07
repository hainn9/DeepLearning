import h5py
import os

class HDF5DatasetWriter:
    def __init__(self, dims, outputPath, dataKey="images", buffSize=1000):
        if os.path.exists(outputPath):
            raise ValueError("The file 'outputPath' already exist and cannnot overwrite", outputPath)

        self.db = h5py.File(outputPath, "w")
        self.data = self.db.create_dataset(dataKey, shape=dims, dtype="float")
        self.labels = self.db.create_dataset("labels", shape=(dims[0],), dtype="int")

        self.buffSize = buffSize
        self.buff = {"data":[], "labels":[]}
        self.idx = 0

    def add(self, row, label):
        self.buff["data"].extend(row)
        self.buff["labels"].extend(label)

        if len(self.buff["data"]) >= self.buffSize:
            self.flush()

    def flush(self):
        i = self.idx + len(self.buff["data"])
        self.data[self.idx:i] = self.buff["data"]
        self.labels[self.idx:i] = self.buff["labels"]
        self.idx = i
        self.buff = {"data":[], "labels":[]}

    def storeClassLabels(self, classLabels):
        dt = h5py.special_dtype(vlen=str)
        labelSet = self.db.create_dataset("label_names", (len(classLabels),), dtype=dt)
        labelSet[:] = classLabels

    def close(self):
        if len(self.buff["data"]) > 0:
            self.flush()
        self.db.close()


        