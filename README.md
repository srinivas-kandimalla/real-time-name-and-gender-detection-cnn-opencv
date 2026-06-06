# Real-Time Name and Gender Detection System using CNN and OpenCV

A computer vision project that detects faces in real-time using a webcam and predicts the person's name and gender using a Convolutional Neural Network (CNN) and OpenCV.

---

## Project Output

![Detection Output](outputs/detection_output.png)

---

## Features

- Real-time face detection
- Name recognition
- Gender prediction
- CNN-based classification
- OpenCV webcam integration
- Confidence score display
- Easy-to-use interface

---

## Technologies Used

- Python
- OpenCV
- TensorFlow / Keras
- NumPy
- CNN (Convolutional Neural Network)

---

## Project Workflow

```text
Face Registration
       ↓
Image Collection
       ↓
CNN Model Training
       ↓
Model Saving
       ↓
Real-Time Webcam Detection
       ↓
Name and Gender Prediction
```

## Project Structure

```text
Real_Time_Name_Gender_Detection/
│
├── dataset/
├── data/
│   └── labels.json
│
├── models/
│   └── face_model.h5
│
├── outputs/
│   └── detection_output.png
│
├── src/
│   ├── register.py
│   ├── train.py
│   └── detect.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

Install required libraries:

```bash
pip install -r requirements.txt
```

## Run the Project

### Step 1: Register Face

```bash
python src/register.py
```

### Step 2: Train Model

```bash
python src/train.py
```

### Step 3: Start Real-Time Detection

```bash
python src/detect.py
```

## Sample Output

```text
Name: Srinivas
Gender: Male
Confidence: 100.0%
```

## Applications

- Face Recognition Systems
- Smart Attendance Systems
- Security and Surveillance
- AI-based Identity Verification

## Future Enhancements

- Multiple user support
- Attendance management
- Database integration
- Improved CNN architecture
- Web-based dashboard

