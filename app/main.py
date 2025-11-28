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

import plotly.graph_objects as go

# Gradio Interface
def gradio_predict(MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude):
    data = [MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]
    pred = model.predict([data])
    
    # Convert to actual dollars
    price_val = pred[0] * 100000
    price_fmt = f"${price_val:,.2f}"
    
    # Create a modern Gauge Chart for the price
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = price_val,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Estimated Value", 'font': {'size': 24, 'color': "#6366f1"}},
        number = {'prefix': "$", 'font': {'size': 40, 'color': "#4ade80"}},
        gauge = {
            'axis': {'range': [0, 500000], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#6366f1"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 150000], 'color': '#e0e7ff'},
                {'range': [150000, 300000], 'color': '#c7d2fe'},
                {'range': [300000, 500000], 'color': '#a5b4fc'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': price_val
            }
        }
    ))
    
    fig.update_layout(paper_bgcolor = "rgba(0,0,0,0)", font = {'color': "gray", 'family': "Arial"})

    return price_fmt, fig

iface = gr.Interface(
    fn=gradio_predict,
    inputs=[
        gr.Slider(minimum=0, maximum=15, step=0.1, label="Median Income (Tens of $1000s)", value=8.3),
        gr.Slider(minimum=1, maximum=52, step=1, label="House Age (Years)", value=41),
        gr.Slider(minimum=1, maximum=20, step=0.1, label="Average Rooms", value=6.9),
        gr.Slider(minimum=0, maximum=10, step=0.1, label="Average Bedrooms", value=1.0),
        gr.Slider(minimum=1, maximum=5000, step=10, label="Population", value=322),
        gr.Slider(minimum=1, maximum=10, step=0.1, label="Average Occupancy", value=2.5),
        gr.Slider(minimum=32, maximum=42, step=0.1, label="Latitude", value=37.88),
        gr.Slider(minimum=-125, maximum=-114, step=0.1, label="Longitude", value=-122.23)
    ],
    outputs=[
        gr.Textbox(label="Predicted Price"),
        gr.Plot(label="Price Gauge")
    ],
    title="🏡 California House Price Predictor",
    description="Adjust the sliders to estimate the house price. The gauge shows the value relative to typical market ranges."
)

# Mount Gradio app to FastAPI
app = gr.mount_gradio_app(app, iface, path="/ui")

@app.get("/")
def read_root():
    return {"message": "Welcome to the House Price Predictor API. Go to /ui for the interface or POST to /predict."}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=10000)
