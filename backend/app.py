from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import pickle
import os

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')  # Optional for EDA

# Load all models and mappings
scaler = pickle.load(open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb"))
kmeans_sleep = pickle.load(open(os.path.join(MODEL_DIR, "kmeans_sleep.pkl"), "rb"))
gmm_sleep = pickle.load(open(os.path.join(MODEL_DIR, "gmm_sleep.pkl"), "rb"))
kmeans_academic = pickle.load(open(os.path.join(MODEL_DIR, "kmeans_academic.pkl"), "rb"))
gmm_academic = pickle.load(open(os.path.join(MODEL_DIR, "gmm_academic.pkl"), "rb"))
sleep_mapping = pickle.load(open(os.path.join(MODEL_DIR, "sleep_mapping.pkl"), "rb"))
academic_mapping = pickle.load(open(os.path.join(MODEL_DIR, "academic_mapping.pkl"), "rb"))
gmm_sleep_mapping = pickle.load(open(os.path.join(MODEL_DIR, "gmm_sleep_mapping.pkl"), "rb"))
gmm_academic_mapping = pickle.load(open(os.path.join(MODEL_DIR, "gmm_academic_mapping.pkl"), "rb"))

sleep_labels = {0: "Night Owl", 1: "Balanced Sleeper", 2: "Oversleeper"}
academic_labels = {0: "Low Performer", 1: "Average Performer", 2: "High Performer"}

def prepare_features(data):
    pa_hr_day = data["Physical_Activity"] / 2 / 60
    return np.array([
        data["Study_Hours"],
        data["Screen_Time"],
        data["Caffeine_Intake"],
        pa_hr_day,
        data["Sleep_Duration"],
        data["Sleep_Quality"]
    ]).reshape(1, -1)

@app.route("/predict/sleep", methods=["POST"])
def predict_sleep():
    data = request.get_json()
    model_type = data.get("model", "KMeans")
    x = scaler.transform(prepare_features(data))
    if model_type == "GMM":
        raw_pred = gmm_sleep.predict(x)[0]
        pred = gmm_sleep_mapping[raw_pred]
        label = sleep_labels.get(pred)
    else:
        raw_pred = kmeans_sleep.predict(x)[0]
        pred = sleep_mapping[raw_pred]
        label = sleep_labels.get(pred)
    return jsonify({"cluster_label": label})

@app.route("/predict/academic", methods=["POST"])
def predict_academic():
    data = request.get_json()
    model_type = data.get("model", "KMeans")
    x = scaler.transform(prepare_features(data))
    if model_type == "GMM":
        raw_pred = gmm_academic.predict(x)[0]
        pred = gmm_academic_mapping[raw_pred]
        label = academic_labels.get(pred)
    else:
        raw_pred = kmeans_academic.predict(x)[0]
        pred = academic_mapping[raw_pred]
        label = academic_labels.get(pred)
    return jsonify({"cluster_label": label})

# Optional EDA endpoint
@app.route("/eda-data", methods=["GET"])
def eda_data():
    try:
        df = pd.read_csv(os.path.join(DATA_DIR, "student_sleep_patterns_updated.csv"))
        df["Physical_Activity"] = df["Physical_Activity"] / 2 / 60
        return df.describe(include="all").to_json()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
