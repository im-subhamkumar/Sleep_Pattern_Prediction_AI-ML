# 🛌🎓 Student Sleep & Academic Performance Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sleep-predictor.streamlit.app)

## 🚀 Live Demo
Access the live application here: **[sleep-predictor.streamlit.app]([https://sleep-predictor.streamlit.app](https://student-sleep-acedmic-analysis.streamlit.app/))**

This project predicts sleep types and academic profiles based on lifestyle details such as study hours, screen time, caffeine intake, and physical activity.

## Features
- **Sleep Type Prediction**: Categorizes users into "Night Owl", "Balanced Sleeper", or "Oversleeper".
- **Academic Profile Prediction**: Categorizes users into "Low", "Average", or "High" performers.
- **Data Visualization**: Explore feature distributions, correlations, and cluster visualizations.

## Tech Stack
- **Frontend**: Streamlit
- **Backend API**: Flask (Deployed on Render)
- **Machine Learning**: KMeans, GMM (Scikit-Learn)

## How to Run Locally

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd Sleep_Pattern_Prediction_AI-ML-main
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**:
   ```bash
   streamlit run streamlit_app.py
   ```

## Deployment
The app is designed to be deployed on [Streamlit Cloud](https://streamlit.io/cloud). 
The backend is currently hosted at: `https://flask-sleep-backend.onrender.com`
