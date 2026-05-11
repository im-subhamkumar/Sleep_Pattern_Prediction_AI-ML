import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.decomposition import PCA

st.set_page_config(page_title="Sleep & Academic Predictor", layout="wide")

st.sidebar.header("Enter Your Lifestyle Details")
study_hours = st.sidebar.slider("Study Hours per day", 0.0, 16.0, 6.0, 0.5)
screen_time = st.sidebar.slider("Screen Time (hours per day)", 0.0, 10.0, 4.0, 0.5)
caffeine = st.sidebar.slider("Caffeine Intake (cups per day)", 0, 8, 2)
activity_daily = st.sidebar.slider("Physical Activity (minutes per day)", 0.0, 180.0, 60.0, 10.0)
sleep_duration = st.sidebar.slider("Sleep Duration (hours per night)", 4.0, 12.0, 7.5, 0.5)
sleep_quality = st.sidebar.slider("Sleep Quality (1-10)", 1, 10, 7)
model_choice = st.sidebar.selectbox("Select Clustering Model", ["KMeans", "GMM"])

# Convert activity to minutes over 2 days as backend expects
activity_total_2days = activity_daily * 2

user_input = {
    "Study_Hours": study_hours,
    "Screen_Time": screen_time,
    "Caffeine_Intake": caffeine,
    "Physical_Activity": activity_total_2days,
    "Sleep_Duration": sleep_duration,
    "Sleep_Quality": sleep_quality,
    "model": model_choice
}

# Update to your deployed backend URL
backend_url = "https://flask-sleep-backend.onrender.com"

sleep_cluster_desc = {
    "Night Owl": "🌙 Less sleep, possible late hours, be cautious about daytime fatigue.",
    "Balanced Sleeper": "✅ Good sleep habits, balanced schedule.",
    "Oversleeper": "😴 Tends to sleep more than average. Monitor for oversleeping or lingering tiredness."
}
academic_cluster_desc = {
    "Low Performer": "⚠️ Current lifestyle may hinder academic progress.",
    "Average Performer": "🟡 On track, but could improve further.",
    "High Performer": "🏆 Excellent! Your habits support strong academic results."
}

st.title("🛌🎓 Student Sleep & Academic Performance Predictor")
col1, col2 = st.columns(2)

with col1:
    st.subheader("😴 Predict My Sleep Type")
    if st.button("Analyze Sleep Habits"):
        try:
            r = requests.post(f"{backend_url}/predict/sleep", json=user_input, timeout=30)
            r.raise_for_status()
            label = r.json()['cluster_label']
            st.success(f"Your Predicted Sleep Type: {label}")
            st.markdown(sleep_cluster_desc.get(label, "Prediction made."))
        except Exception as e:
            st.error(f"Error: {e}")

with col2:
    st.subheader("🎓 Predict My Academic Profile")
    if st.button("Analyze Academic Profile"):
        try:
            r = requests.post(f"{backend_url}/predict/academic", json=user_input, timeout=30)
            r.raise_for_status()
            label = r.json()['cluster_label']
            st.success(f"Your Predicted Academic Profile: {label}")
            st.markdown(academic_cluster_desc.get(label, "Prediction made."))
        except Exception as e:
            st.error(f"Error: {e}")

st.markdown("---")
st.subheader("📉 Cluster Visualization")
viz_choice = st.radio("Choose Cluster to Visualize:", ["Sleep Behavior", "Academic Performance"], horizontal=True)

if st.checkbox("Show Selected Cluster Visualization"):
    try:
        model_folder = "models"
        if viz_choice == "Sleep Behavior":
            model_fname = f"{model_folder}/kmeans_sleep.pkl" if model_choice == "KMeans" else f"{model_folder}/gmm_sleep.pkl"
            map_fname = f"{model_folder}/sleep_mapping.pkl" if model_choice == "KMeans" else f"{model_folder}/gmm_sleep_mapping.pkl"
        else:
            model_fname = f"{model_folder}/kmeans_academic.pkl" if model_choice == "KMeans" else f"{model_folder}/gmm_academic.pkl"
            map_fname = f"{model_folder}/academic_mapping.pkl" if model_choice == "KMeans" else f"{model_folder}/gmm_academic_mapping.pkl"

        scaler_fname = f"{model_folder}/scaler.pkl"

        with open(model_fname, "rb") as f:
            model = pickle.load(f)
        with open(map_fname, "rb") as f:
            mapping = pickle.load(f)
        with open(scaler_fname, "rb") as f:
            scaler = pickle.load(f)

        df = pd.read_csv("data/student_sleep_patterns_updated.csv")
        df["Physical_Activity"] = df["Physical_Activity"] / 2 / 60
        features = [
            "Study_Hours", "Screen_Time", "Caffeine_Intake",
            "Physical_Activity", "Sleep_Duration", "Sleep_Quality"
        ]
        data = df[features].dropna()
        data_scaled = scaler.transform(data)
        clusters = model.predict(data_scaled)
        mapped_clusters = [mapping.get(c, c) for c in clusters]
        pca = PCA(n_components=2)
        data_pca = pca.fit_transform(data_scaled)

        viz_df = pd.DataFrame(data_pca, columns=["PCA1", "PCA2"])
        viz_df["Cluster"] = [str(c) for c in mapped_clusters]

        user_feats = [
            user_input["Study_Hours"],
            user_input["Screen_Time"],
            user_input["Caffeine_Intake"],
            user_input["Physical_Activity"] / 2 / 60,
            user_input["Sleep_Duration"],
            user_input["Sleep_Quality"]
        ]
        user_scaled = scaler.transform([user_feats])
        user_pca = pca.transform(user_scaled)

        fig, ax = plt.subplots(figsize=(7, 5))
        palette = sns.color_palette("Set2", n_colors=len(set(mapped_clusters)))
        sns.scatterplot(
            x="PCA1", y="PCA2", hue="Cluster", data=viz_df, ax=ax,
            s=80, alpha=0.85, legend="full", palette=palette
        )
        ax.scatter(user_pca[0, 0], user_pca[0, 1], color="black", s=200, marker="X", label="You")
        ax.set_title(f"{viz_choice} Clusters ({model_choice})")
        handles, labels = ax.get_legend_handles_labels()
        if "You" not in labels:
            handles.append(ax.scatter([], [], color="black", marker="X", s=200))
            labels.append("You")
        ax.legend(handles, labels, title="Cluster")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Visualization error: {e}")

st.markdown("---")
st.header("📊 Explore the Data")

if st.checkbox("Show Data Insights and Visualizations"):
    try:
        df = pd.read_csv("data/student_sleep_patterns_updated.csv")
        df["Physical_Activity"] = df["Physical_Activity"] / 2 / 60
        features = [
            "Study_Hours", "Screen_Time", "Caffeine_Intake",
            "Physical_Activity", "Sleep_Duration", "Sleep_Quality"
        ]

        st.subheader("Feature Distributions")
        fig_hist, axes_hist = plt.subplots(2, 3, figsize=(15, 8))
        df[features].hist(ax=axes_hist.flatten()[:len(features)], bins=15)
        fig_hist.tight_layout()
        st.pyplot(fig_hist)

        st.subheader("Correlation Between Features")
        fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
        sns.heatmap(df[features].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax_corr)
        st.pyplot(fig_corr)

        st.subheader("Elbow Method for Optimal Clusters (KMeans)")
        scaler = pickle.load(open("models/scaler.pkl", "rb"))
        data_scaled = scaler.transform(df[features].dropna())
        max_clusters = min(10, len(data_scaled))
        sse = []
        if max_clusters > 2:
            from sklearn.cluster import KMeans
            for k in range(2, max_clusters):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(data_scaled)
                sse.append(kmeans.inertia_)
            fig_elbow, ax_elbow = plt.subplots()
            ax_elbow.plot(range(2, max_clusters), sse, marker='o')
            ax_elbow.set_xlabel("Number of Clusters (k)")
            ax_elbow.set_ylabel("Sum of Squared Errors (SSE)")
            ax_elbow.set_title("Elbow Method")
            st.pyplot(fig_elbow)
        else:
            st.warning("Not enough data to perform Elbow Method analysis.")
    except Exception as e:
        st.error(f"An error occurred during visualization: {e}")
