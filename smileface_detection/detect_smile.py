from keras.preprocessing.image import img_to_array
from keras.models import load_model

import cv2
import argparse
import numpy as np 
import imutils

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="Path of input model")
ap.add_argument("-c", "--cascade", required=True, help="Path to the cascade")
ap.add_argument("-v", "--video", help="Video file") 
args = vars(ap.parse_args())

model = load_model(args["model"])
detector = cv2.CascadeClassifier(args["cascade"])
if not args.get(args["video"], False):
    camera = cv2.VideoCapture(0)
else:
    camera = cv2.VideoCapture(args["video"])

while True:
    (grabbed, frame) = camera.read()
    if args.get("video") and not grabbed:
        break

    frame = imutils.resize(frame, width=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frameClone = frame.copy()

    rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    for (fX, fY, fW, fH) in rects:
        roi = gray[fY:fY+fH, fX:fX+fW]
        roi = cv2.resize(roi, (28, 28))
        roi = roi.stype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        (notSmiling, smiling) = model.predict(roi)[0]
        label = "Smiling" if smiling > notSmiling else "Not_Smiling"

        cv2.putText(frameClone, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(frameClone, (fX, fY), (fX+fW, fY+fH), (0, 0, 255), 2)

    cv2.imshow("Face", frameClone)
    if cv2.waitKey(1) & (0xFF == ord("q")):
        break
camera.release()
cv2.destroyAllWindows()
