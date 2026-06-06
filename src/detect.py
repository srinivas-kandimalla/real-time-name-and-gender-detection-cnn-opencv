import cv2
import json
import numpy as np

from tensorflow.keras.models import load_model

# Load trained model
model = load_model("models/face_model.h5")

# Load label mapping
with open("data/labels.json", "r") as f:
    labels = json.load(f)

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

# Start webcam
cap = cv2.VideoCapture(0)

print("Press ESC to exit")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(100, 100)
    )

    for (x, y, w, h) in faces:

        face = gray[y:y+h, x:x+w]

        face = cv2.resize(
            face,
            (100, 100)
        )

        face = face.reshape(
            1,
            100,
            100,
            1
        )

        face = face / 255.0

        prediction = model.predict(
            face,
            verbose=0
        )

        label_id = np.argmax(
            prediction
        )

        person = labels[str(label_id)]

        # Folder name format:
        # Srinivas_Male
        try:
            name, gender = person.split("_")
        except:
            name = person
            gender = "Unknown"

        confidence = np.max(
            prediction
        ) * 100

        display_text = (
            f"Name: {name} | "
            f"Gender: {gender}"
        )

        cv2.rectangle(
            frame,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            display_text,
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"{confidence:.1f}%",
            (x, y+h+25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

    cv2.imshow(
        "Real-Time Name and Gender Detection",
        frame
    )

    key = cv2.waitKey(1)

    if key == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()