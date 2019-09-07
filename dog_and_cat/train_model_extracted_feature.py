from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import argparse
import h5py
import pickle

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--database", required=True, help="Path of input feature hdf5 files")
ap.add_argument("-m", "--model", required=True, help="Path of output model")
ap.add_argument("-j", "--job", type=int, default=-1, help="Jobs number when turn parameters")
args = vars(ap.parse_args())

db = h5py.File(args["database"], "r")
i = int(db["labels"].shape[0]*0.75)

print("[INFO] Turning parameters for LogisticRegression")
params = {"C" : [0.0001, 0.001, 0.01, 0.1, 1.0]}
# Fix warning FutureWarning for LogisticRegression solver = liblinear -> lbfgs and multi_class = ovr -> auto
# Fix warning ConvergenceWarning lbfgs failed to convergence by increase the number of iterations to 1000
model = GridSearchCV(LogisticRegression(solver="lbfgs", multi_class="auto", max_iter=1000), params, cv=3, n_jobs=args["job"])
model.fit(db["features"][:i], db["labels"][:i])
print("[INFO] Best parameter is {}".format(model.best_params_))

print("[INFO] Evaluating model")
predictions = model.predict(db["features"][i:])
print(classification_report(db["labels"][i:], predictions, target_names=db["label_names"]))

acc = accuracy_score(db["labels"][i:], predictions)
print("[INFO] accuracy score : {}".format(acc))

print("[INFO] Saving model")
f = open(args["model"], "wb")
f.write(pickle.dumps(model.best_estimator_))
f.close()

db.close()
