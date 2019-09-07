##-------------------
##
## Run CMD: python emotion_detector.py --cascade haarcascade_frontalface_default.xml --model checkpoints/epoch_75.hdf5
##
##-------------------
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True, help="Path of input cascade file")
ap.add_argument("-m", "--model", required=True, help="Path of pre-trained model")
ap.add_argument("-v", "--video", help="Path of input video")
args = vars(ap.parse_args())

detector = cv2.CascadeClassifier(args["cascade"])
model = load_model(args["model"])
EMOTIONS = ["angry", "scared", "happy", "sad", "surprised", "neutral"]

if not args.get("video", False):
    camera = cv2.VideoCapture(1)
else:
    camera = cv2.VideoCapture(args["video"])

while True:
    (grabbed, frame) = camera.read()
    if args.get("video") and not grabbed:
        break

    frame = imutils.resize(frame, width=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    canvas = np.zeros((220, 300, 3), dtype="uint8")
    frameClone = frame.copy()

    rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    if len(rects) > 0:
        rect = sorted(rects, reverse=True, key= lambda x:(x[2] - x[0])*(x[3] - x[1]))[0]
        (fX, fY, fW, fH) = rect

        roi = gray[fX:fX + fW, fY:fY + fH]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float")/255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        preds = model.predict(roi)[0]
        label = EMOTIONS[preds.argmax()]
        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
            text = "{} : {:.2f}%".format(emotion, prob*100)
            w = int(prob*100)
            cv2.rectangle(canvas, (5, (i * 35) + 5), (w, (i * 35) + 35), (0, 0, 255), -1)
            cv2.putText(canvas, text, (10, (i * 35) + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
        cv2.putText(frameClone, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)

    cv2.imshow("Face", frameClone)
    cv2.imshow("Probabilities", canvas)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
