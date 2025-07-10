from fastapi import FastAPI
import joblib
import cv2
import keras
from typing import Annotated
import  tensorflow
from fastapi import FastAPI, File, UploadFile
import numpy as np

model = joblib.load("model.pkl")


app = FastAPI(
        title = "API for recognition of images",

)

@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}


@app.post("/predict")
async def predict(file: Annotated[bytes, File()]):
    nparr = np.frombuffer(file, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Некорректный формат изображения"}
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    result = model.predict(img)
    class_names = [
        "daisy - маргаритка",
        "dandelion - одуванчик",
        "rose - роза",
        "sunflower - подсолнух",
        "tulip - тюльпан"
    ]
    pred_index = np.argmax(result[0])
    class_flower = class_names[pred_index]
    print(class_flower)
    return {"result": class_flower}




