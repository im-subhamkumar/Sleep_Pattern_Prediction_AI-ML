import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

print("Starting model training process...")

# Load and preprocess
df = pd.read_csv("data/student_sleep_patterns_updated.csv")
df["Physical_Activity"] = df["Physical_Activity"] / 2 / 60  # mins/2days → hours/day

features = [
    "Study_Hours", "Screen_Time", "Caffeine_Intake",
    "Physical_Activity", "Sleep_Duration", "Sleep_Quality"
]
data = df[features].dropna()

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

kmeans_sleep = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_sleep.fit(data_scaled)
gmm_sleep = GaussianMixture(n_components=3, random_state=42)
gmm_sleep.fit(data_scaled)

kmeans_academic = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_academic.fit(data_scaled)
gmm_academic = GaussianMixture(n_components=3, random_state=42)
gmm_academic.fit(data_scaled)

def get_ordered_mapping(model, feature_name, feature_list):
    idx = feature_list.index(feature_name)
    ordered = np.argsort(model.cluster_centers_[:, idx])
    return {orig: new for new, orig in enumerate(ordered)}

def get_gmm_ordered_mapping(gmm, data_scaled, data_df, feature_name):
    preds = gmm.predict(data_scaled)
    means = []
    for i in range(gmm.n_components):
        means.append((i, data_df.iloc[(preds == i)][feature_name].mean()))
    means.sort(key=lambda x: x[1])
    return {orig: new for new, (orig, _) in enumerate(means)}

sleep_mapping = get_ordered_mapping(kmeans_sleep, 'Sleep_Duration', features)
academic_mapping = get_ordered_mapping(kmeans_academic, 'Study_Hours', features)
gmm_sleep_mapping = get_gmm_ordered_mapping(gmm_sleep, data_scaled, data, "Sleep_Duration")
gmm_academic_mapping = get_gmm_ordered_mapping(gmm_academic, data_scaled, data, "Study_Hours")

os.makedirs("models", exist_ok=True)
pickle.dump(scaler, open("models/scaler.pkl", "wb"))
pickle.dump(kmeans_sleep, open("models/kmeans_sleep.pkl", "wb"))
pickle.dump(gmm_sleep, open("models/gmm_sleep.pkl", "wb"))
pickle.dump(kmeans_academic, open("models/kmeans_academic.pkl", "wb"))
pickle.dump(gmm_academic, open("models/gmm_academic.pkl", "wb"))
pickle.dump(sleep_mapping, open("models/sleep_mapping.pkl", "wb"))
pickle.dump(academic_mapping, open("models/academic_mapping.pkl", "wb"))
pickle.dump(gmm_sleep_mapping, open("models/gmm_sleep_mapping.pkl", "wb"))
pickle.dump(gmm_academic_mapping, open("models/gmm_academic_mapping.pkl", "wb"))

print("✅ Models and mappings successfully trained and saved to /models.")
