from fastapi import FastAPI, UploadFile, File, Form, WebSocket
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import base64
import cv2
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: specify allowed origins (good for production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Item(BaseModel):
    name: str
    price: float
    is_offer: bool | None = None

class Frame(BaseModel):
    image: str


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: str | None = None):
    return {"item_id": item_id, "q": q}


@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    return {"item_name": item.name, "item_id": item_id}

@app.post("/calibrate")
async def calibrate(
    files: List[UploadFile] = File(...),
    labels: List[str] = Form(...)
):
    data = {}

    for file, label in zip(files, labels):
        contents = await file.read()

        if label not in data:
            data[label] = []

        data[label].append(contents)

    print({k: len(v) for k, v in data.items()})

    return {"status": "received"}

@app.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket):
    await websocket.accept()

    try:
        while True:
            data = await websocket.receive_text()

            # remove prefix if exists
            if "," in data:
                data = data.split(",")[1]

            img_bytes = base64.b64decode(data)

            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # TODO: replace with your ML model
            prediction = random.choice(["left", "right", "up", "down", "closed", "open"])  # dummy

            await websocket.send_json({
                "direction": prediction
            })

    except Exception as e:
        print("WebSocket closed:", e)

@app.post("/predict")
def predict(frame: Frame):
    # decode base64
    img_data = base64.b64decode(frame.image.split(",")[1])
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # TODO: replace with your ML model
    direction = dummy_model(img)

    return {"direction": direction}

## TODO: replace with actual ML model
import random

def dummy_model(img):
    return random.choice(["left", "right", "up", "down", "closed", "open"])