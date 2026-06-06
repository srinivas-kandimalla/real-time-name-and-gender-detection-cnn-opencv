import cv2
import os

name = input("Enter Name: ")
gender = input("Enter Gender: ")

folder_name = f"dataset/{name}_{gender}"
os.makedirs(folder_name, exist_ok=True)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

count = 0

print("Capturing high-quality face images...")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(100, 100)
    )

    for (x, y, w, h) in faces:

        padding = 20

        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)

        face = gray[y1:y2, x1:x2]

        # Better image size
        face = cv2.resize(face, (128, 128))

        # Improve brightness
        face = cv2.equalizeHist(face)

        count += 1

        cv2.imwrite(
            f"{folder_name}/{count}.jpg",
            face
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Images: {count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    cv2.imshow("Face Registration", frame)

    key = cv2.waitKey(1)

    if key == 27:
        break

    if count >= 100:
        break

cap.release()
cv2.destroyAllWindows()

print("Dataset created successfully!")