from fastapi import FastAPI, UploadFile, File
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.applications.resnet50 import preprocess_input

app = FastAPI()

model = tf.keras.models.load_model("face_mask_model.keras")

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

@app.post("/image_predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()

    image = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, (224, 224))

    image = preprocess_input(image.astype(np.float32))

    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)

    probability = float(prediction[0][0])

    if probability < 0.5:
        label = "MASK ON"
    else:
        label = "MASK OFF"

    return {
        "prediction": label,
        "confidence": probability
    }

@app.post("/vid_predict")
async def vid_predict(file: UploadFile = File(...)):
    # Read uploaded frame
    image_bytes = await file.read()

    image_array = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if frame is None:
        return {
            "prediction": "Invalid image",
            "confidence": 0
        }

    # Detect face
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    coords = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=8
    )

    if len(coords) == 0:
        return {
            "prediction": "No face detected",
            "confidence": 0
        }

    # Use first detected face
    x, y, w, h = coords[0]

    face = frame[y:y+h, x:x+w]

    # Preprocess
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (224,224))
    face = preprocess_input(face.astype(np.float32))
    face = np.expand_dims(face, axis=0)

    prediction = model.predict(face, verbose=0)

    probability = float(prediction[0][0])

    if probability < 0.5:
        label = "MASK ON"
    else:
        label = "MASK OFF"

    return {
        "prediction": label,
        "confidence": probability,
        "bbox": [int(x), int(y), int(w), int(h)]
    }