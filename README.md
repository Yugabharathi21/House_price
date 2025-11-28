# 🏠 House Price Prediction App

A machine learning-powered web application to predict California house prices. This project combines a **FastAPI** backend for high-performance API endpoints and a **Gradio** frontend for an interactive user interface.

## 🚀 Features

- **FastAPI Backend**: Robust REST API for making predictions programmatically.
- **Gradio UI**: User-friendly web interface to input housing details and get instant predictions.
- **Machine Learning**: Uses a Linear Regression model trained on the California Housing dataset.
- **Ready for Deployment**: Configured for easy deployment on Render.com.

## 📂 Project Structure

```
.
├── app/
│   ├── main.py          # FastAPI application with Gradio mounted
│   └── house_model.pkl  # Pre-trained machine learning model
├── scripts/
│   └── train_model.py   # Script to retrain the model
├── requirements.txt     # Python dependencies
├── render.yaml          # Render deployment configuration
└── README.md            # Project documentation
```

## 🛠️ Local Setup

Follow these steps to run the application on your local machine.

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/house-price-api.git
cd house-price-api
```

### 2. Install Dependencies
It is recommended to use a virtual environment.
```bash
pip install -r requirements.txt
```

### 3. (Optional) Retrain the Model
If you want to regenerate the model file:
```bash
python scripts/train_model.py
```
This will save a new `house_model.pkl` in the `app/` directory.

### 4. Run the Application
Start the server using Uvicorn:
```bash
uvicorn app.main:app --reload
```

- **API Docs**: Visit `http://127.0.0.1:8000/docs` to see the Swagger UI.
- **Gradio UI**: Visit `http://127.0.0.1:8000/ui` to use the interactive interface.

---

## ☁️ How to Host on Render

This project is configured for seamless deployment on [Render](https://render.com).

### Option 1: Automatic Deployment (Blueprints) - Recommended

1.  **Push to GitHub**: Ensure your code is pushed to a GitHub repository.
2.  **Log in to Render**: Go to [dashboard.render.com](https://dashboard.render.com/).
3.  **New Blueprint**: Click **New +** and select **Blueprint**.
4.  **Connect Repo**: Connect your GitHub repository.
5.  **Deploy**: Render will automatically detect the `render.yaml` file and configure the service for you.
6.  **Done!**: Your app will be live at `https://your-app-name.onrender.com`.

### Option 2: Manual Setup

1.  **Log in to Render**: Go to [dashboard.render.com](https://dashboard.render.com/).
2.  **New Web Service**: Click **New +** and select **Web Service**.
3.  **Connect Repo**: Connect your GitHub repository.
4.  **Configure Settings**:
    - **Name**: `house-price-predictor` (or any name you like)
    - **Runtime**: `Python 3`
    - **Build Command**: `pip install -r requirements.txt`
    - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port 10000`
5.  **Deploy**: Click **Create Web Service**.

### Accessing Your Deployed App

Once deployed:
- **API**: `https://your-app-name.onrender.com/predict` (POST request)
- **UI**: `https://your-app-name.onrender.com/ui` (Interactive Interface)
