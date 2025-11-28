from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import joblib
import gradio as gr
import uvicorn

import os

app = FastAPI()
model_path = os.path.join(os.path.dirname(__file__), "house_model.pkl")
model = joblib.load(model_path)

class Input(BaseModel):
    data: Optional[list] = [8.3252, 41.0, 6.98, 1.02, 322, 2.55, 37.88, -122.23]

@app.post("/predict")
def predict(input: Input = Input()):
    pred = model.predict([input.data])
    return {"prediction": pred[0]}

# Gradio Interface
def gradio_predict(MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude):
    data = [MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]
    pred = model.predict([data])
    return pred[0]

iface = gr.Interface(
    fn=gradio_predict,
    inputs=[
        gr.Number(label="Median Income", value=8.3252),
        gr.Number(label="House Age", value=41.0),
        gr.Number(label="Average Rooms", value=6.98),
        gr.Number(label="Average Bedrooms", value=1.02),
        gr.Number(label="Population", value=322),
        gr.Number(label="Average Occupancy", value=2.55),
        gr.Number(label="Latitude", value=37.88),
        gr.Number(label="Longitude", value=-122.23)
    ],
    outputs="number",
    title="California House Price Predictor",
    description="Enter the housing details to predict the price."
)

# Mount Gradio app to FastAPI
app = gr.mount_gradio_app(app, iface, path="/ui")

@app.get("/")
def read_root():
    return {"message": "Welcome to the House Price Predictor API. Go to /ui for the interface or POST to /predict."}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=10000)
