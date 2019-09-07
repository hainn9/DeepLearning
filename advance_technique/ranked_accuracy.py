from dl_cv.utils.ranked import rank5_accuracy
import pickle
import argparse
import h5py

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--database", required=True, help="Path of input database hdf5 file")
ap.add_argument("-m", "--model", required=True, help="Path of pickle model")
args = vars(ap.parse_args())

print("[INFO] Load pre-trained model")
model = pickle.loads(open(args["model"], "rb").read())

db = h5py.File(args["database"], "r")
i = int(db["labels"].shape[0]*0.75)

print("[INFO] Calculate accuracy")
predictions = model.predict_proba(db["features"][i:])
(rank1, rank5) = rank5_accuracy(predictions, db["labels"][i:])

print("[INFO] Rank-1 accuracy: {:.2f}%".format(rank1*100))
print("[INFO] Rank-5 accuracy: {:.2f}%".format(rank5*100))
db.close()