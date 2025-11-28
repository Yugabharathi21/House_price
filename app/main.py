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

import pandas as pd
from datetime import datetime
import plotly.graph_objects as go

# Global history
history = []

def predict_and_log(MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude):
    # 1. Predict
    data = [MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]
    pred = model.predict([data])
    price_val = pred[0] * 100000
    price_fmt = f"${price_val:,.2f}"
    
    # 2. Charts (Bar + Map)
    # ... (Bar Chart Logic)
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(x=[500000], y=[""], orientation='h', marker=dict(color='rgba(0,0,0,0.05)'), hoverinfo='skip', width=0.4))
    fig_bar.add_trace(go.Bar(x=[price_val], y=[""], orientation='h', marker=dict(color='#6366f1'), hoverinfo='skip', width=0.4))
    fig_bar.update_layout(
        barmode='overlay', xaxis=dict(range=[0, 500000], showgrid=False, showticklabels=False, fixedrange=True),
        yaxis=dict(showgrid=False, showticklabels=False, fixedrange=True), showlegend=False, height=150,
        margin=dict(l=0, r=0, t=40, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        annotations=[
            dict(x=price_val, y=0, text=price_fmt, showarrow=True, arrowhead=0, ax=0, ay=-50, font=dict(size=32, color="#6366f1", weight="bold")),
            dict(x=0, y=0, text="$0", showarrow=False, yshift=-30, font=dict(size=12, color="gray")),
            dict(x=500000, y=0, text="$500k+", showarrow=False, xanchor="right", yshift=-30, font=dict(size=12, color="gray"))
        ]
    )

    # ... (Map Logic)
    fig_map = go.Figure(go.Scattermapbox(
        lat=[Latitude], lon=[Longitude], mode='markers',
        marker=go.scattermapbox.Marker(size=14, color='red'), text=[f"Price: {price_fmt}"], hoverinfo='text'
    ))
    fig_map.update_layout(
        mapbox_style="open-street-map", hovermode='closest',
        mapbox=dict(center=go.layout.mapbox.Center(lat=Latitude, lon=Longitude), zoom=10),
        margin={"r":0,"t":0,"l":0,"b":0}, height=300
    )

    # 3. Log to History
    status = "🚩 High Value" if price_val > 400000 else "Standard"
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    new_entry = {
        "Time": timestamp,
        "Price": price_fmt,
        "Status": status,
        "Income": MedInc,
        "Age": HouseAge,
        "Rooms": AveRooms
    }
    history.insert(0, new_entry) # Add to top
    
    return price_fmt, fig_bar, fig_map, pd.DataFrame(history)

# Modern UI Layout with Blocks
with gr.Blocks(title="California House Price Predictor") as demo:
    gr.Markdown("""
    # 🏡 California House Price Predictor
    Adjust the sliders to estimate house prices. High-value properties (>$400k) are flagged in the history.
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            inp_inc = gr.Slider(0, 15, 8.3, step=0.1, label="Median Income (Tens of $1000s)")
            inp_age = gr.Slider(1, 52, 41, step=1, label="House Age (Years)")
            inp_rooms = gr.Slider(1, 20, 6.9, step=0.1, label="Average Rooms")
            inp_bedrms = gr.Slider(0, 10, 1.0, step=0.1, label="Average Bedrooms")
            inp_pop = gr.Slider(1, 5000, 322, step=10, label="Population")
            inp_occup = gr.Slider(1, 10, 2.5, step=0.1, label="Average Occupancy")
            inp_lat = gr.Slider(32.5, 42.0, 37.88, step=0.01, label="Latitude (CA Region)")
            inp_lon = gr.Slider(-124.35, -114.1, -122.23, step=0.01, label="Longitude (CA Region)")
            
            btn = gr.Button("Predict Price", variant="primary")

        with gr.Column(scale=2):
            out_price = gr.Textbox(label="Predicted Price")
            out_plot = gr.Plot(label="Price Gauge")
            out_map = gr.Plot(label="Location Map")
    
    with gr.Row():
        out_table = gr.Dataframe(label="Prediction History (Flagged)", headers=["Time", "Price", "Status", "Income", "Age", "Rooms"])

    # Link inputs to function
    btn.click(
        fn=predict_and_log,
        inputs=[inp_inc, inp_age, inp_rooms, inp_bedrms, inp_pop, inp_occup, inp_lat, inp_lon],
        outputs=[out_price, out_plot, out_map, out_table]
    )

# Mount Gradio app to FastAPI
app = gr.mount_gradio_app(app, demo, path="/ui")

@app.get("/")
def read_root():
    return {"message": "Welcome to the House Price Predictor API. Go to /ui for the interface or POST to /predict."}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=10000)
