import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ─── Page Config ───
st.set_page_config(
    page_title="Sleep & Academic Predictor",
    page_icon="🛌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* Global */
html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1.5rem; }

/* Header */
.hero-title {
    font-size: 2.6rem; font-weight: 800;
    background: linear-gradient(135deg, #6C63FF 0%, #E040FB 50%, #FF6584 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem; line-height: 1.2;
}
.hero-sub { color: #9e9e9e; font-size: 1.05rem; margin-bottom: 1.5rem; }

/* Glass cards */
.glass-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px; padding: 1.6rem;
    backdrop-filter: blur(12px);
    transition: transform 0.25s, box-shadow 0.25s;
}
.glass-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 32px rgba(108,99,255,0.15);
}

/* Result cards */
.result-card {
    background: linear-gradient(135deg, rgba(108,99,255,0.12), rgba(224,64,251,0.08));
    border: 1px solid rgba(108,99,255,0.25);
    border-radius: 14px; padding: 1.4rem; text-align: center;
}
.result-label { font-size: 1.6rem; font-weight: 700; color: #E0E0E0; margin: 0.4rem 0; }
.result-desc { font-size: 0.95rem; color: #BDBDBD; }

/* Metric cards */
.metric-row { display: flex; gap: 0.8rem; flex-wrap: wrap; margin: 1rem 0; }
.metric-card {
    flex: 1; min-width: 120px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px; padding: 1rem; text-align: center;
}
.metric-val { font-size: 1.5rem; font-weight: 700; color: #6C63FF; }
.metric-lbl { font-size: 0.78rem; color: #9e9e9e; margin-top: 0.2rem; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
}
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stSelectbox label { color: #E0E0E0 !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 1.5rem; }
.stTabs [data-baseweb="tab"] {
    font-weight: 600; color: #9e9e9e;
    border-bottom: 2px solid transparent; padding-bottom: 0.5rem;
}
.stTabs [aria-selected="true"] {
    color: #6C63FF !important;
    border-bottom-color: #6C63FF !important;
}

/* Divider */
.section-divider {
    height: 1px; margin: 2rem 0;
    background: linear-gradient(90deg, transparent, rgba(108,99,255,0.3), transparent);
}
</style>
""", unsafe_allow_html=True)

# ─── Load Models (cached) ───
@st.cache_resource
def load_models():
    base = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base, "models")
    m = {}
    for name in [
        "scaler", "kmeans_sleep", "gmm_sleep", "kmeans_academic", "gmm_academic",
        "sleep_mapping", "academic_mapping", "gmm_sleep_mapping", "gmm_academic_mapping",
    ]:
        with open(os.path.join(model_dir, f"{name}.pkl"), "rb") as f:
            m[name] = pickle.load(f)
    return m

@st.cache_data
def load_data():
    base = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(base, "data", "student_sleep_patterns_updated.csv"))
    df["Physical_Activity"] = df["Physical_Activity"] / 2 / 60
    return df

models = load_models()
df = load_data()

FEATURES = ["Study_Hours", "Screen_Time", "Caffeine_Intake",
            "Physical_Activity", "Sleep_Duration", "Sleep_Quality"]
SLEEP_LABELS = {0: "Night Owl", 1: "Balanced Sleeper", 2: "Oversleeper"}
ACADEMIC_LABELS = {0: "Low Performer", 1: "Average Performer", 2: "High Performer"}

SLEEP_DESC = {
    "Night Owl": ("🌙", "Less sleep, possible late hours — be cautious about daytime fatigue.",
                  ["Aim for 7-8 hours of sleep", "Avoid screens 1hr before bed", "Try a fixed wake-up time"]),
    "Balanced Sleeper": ("✅", "Great sleep habits and a balanced schedule. Keep it up!",
                         ["Maintain your routine", "Stay consistent on weekends", "Keep caffeine moderate"]),
    "Oversleeper": ("😴", "Sleeping more than average — monitor for lingering tiredness.",
                    ["Set a consistent alarm", "Increase morning physical activity", "Check iron/vitamin D levels"]),
}
ACADEMIC_DESC = {
    "Low Performer": ("⚠️", "Current lifestyle patterns may be hindering academic progress.",
                      ["Increase focused study blocks", "Reduce screen time", "Improve sleep consistency"]),
    "Average Performer": ("🟡", "You're on track, but there's room for improvement.",
                          ["Add 1 extra study hour", "Try active recall techniques", "Optimize sleep schedule"]),
    "High Performer": ("🏆", "Excellent — your habits strongly support academic success!",
                       ["Maintain current balance", "Mentor others", "Explore advanced learning"]),
}

# ─── Prediction Functions ───
def prepare_features(data):
    return np.array([[data["Study_Hours"], data["Screen_Time"], data["Caffeine_Intake"],
                      data["Physical_Activity"] / 2 / 60, data["Sleep_Duration"], data["Sleep_Quality"]]])

def predict_sleep(data, model_type):
    x = models["scaler"].transform(prepare_features(data))
    if model_type == "GMM":
        raw = models["gmm_sleep"].predict(x)[0]
        mapped = models["gmm_sleep_mapping"][raw]
    else:
        raw = models["kmeans_sleep"].predict(x)[0]
        mapped = models["sleep_mapping"][raw]
    return SLEEP_LABELS[mapped]

def predict_academic(data, model_type):
    x = models["scaler"].transform(prepare_features(data))
    if model_type == "GMM":
        raw = models["gmm_academic"].predict(x)[0]
        mapped = models["gmm_academic_mapping"][raw]
    else:
        raw = models["kmeans_academic"].predict(x)[0]
        mapped = models["academic_mapping"][raw]
    return ACADEMIC_LABELS[mapped]

# ─── Sidebar ───
with st.sidebar:
    st.markdown("### ⚙️ Your Lifestyle")
    st.caption("Adjust the sliders to match your daily habits")
    study_hours = st.slider("📚 Study Hours / day", 0.0, 16.0, 6.0, 0.5)
    screen_time = st.slider("📱 Screen Time (hrs/day)", 0.0, 10.0, 4.0, 0.5)
    caffeine = st.slider("☕ Caffeine (cups/day)", 0, 8, 2)
    activity = st.slider("🏃 Physical Activity (min/day)", 0.0, 180.0, 60.0, 10.0)
    sleep_dur = st.slider("🛏️ Sleep Duration (hrs/night)", 4.0, 12.0, 7.5, 0.5)
    sleep_qual = st.slider("⭐ Sleep Quality (1-10)", 1, 10, 7)
    model_choice = st.selectbox("🤖 Clustering Model", ["KMeans", "GMM"])
    st.markdown("---")
    st.caption("Built with Streamlit • scikit-learn")

user_input = {
    "Study_Hours": study_hours, "Screen_Time": screen_time,
    "Caffeine_Intake": caffeine, "Physical_Activity": activity * 2,
    "Sleep_Duration": sleep_dur, "Sleep_Quality": sleep_qual,
}

# ─── Hero Header ───
st.markdown('<p class="hero-title">Student Sleep & Academic Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Discover your sleep type and academic profile using AI clustering models trained on real student data.</p>', unsafe_allow_html=True)

# ─── Your Profile Metrics ───
st.markdown('<div class="metric-row">'
    f'<div class="metric-card"><div class="metric-val">{study_hours}h</div><div class="metric-lbl">Study</div></div>'
    f'<div class="metric-card"><div class="metric-val">{screen_time}h</div><div class="metric-lbl">Screen</div></div>'
    f'<div class="metric-card"><div class="metric-val">{caffeine}</div><div class="metric-lbl">Caffeine</div></div>'
    f'<div class="metric-card"><div class="metric-val">{activity:.0f}m</div><div class="metric-lbl">Activity</div></div>'
    f'<div class="metric-card"><div class="metric-val">{sleep_dur}h</div><div class="metric-lbl">Sleep</div></div>'
    f'<div class="metric-card"><div class="metric-val">{sleep_qual}/10</div><div class="metric-lbl">Quality</div></div>'
    '</div>', unsafe_allow_html=True)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ─── Predictions ───
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 😴 Sleep Type Analysis")
    if st.button("🔍 Analyze Sleep Habits", width='stretch'):
        label = predict_sleep(user_input, model_choice)
        emoji, desc, tips = SLEEP_DESC[label]
        st.markdown(f'<div class="result-card">'
            f'<div style="font-size:2.5rem">{emoji}</div>'
            f'<div class="result-label">{label}</div>'
            f'<div class="result-desc">{desc}</div></div>', unsafe_allow_html=True)
        st.markdown("**💡 Recommendations:**")
        for tip in tips:
            st.markdown(f"- {tip}")

with col2:
    st.markdown("#### 🎓 Academic Profile Analysis")
    if st.button("🔍 Analyze Academic Profile", width='stretch'):
        label = predict_academic(user_input, model_choice)
        emoji, desc, tips = ACADEMIC_DESC[label]
        st.markdown(f'<div class="result-card">'
            f'<div style="font-size:2.5rem">{emoji}</div>'
            f'<div class="result-label">{label}</div>'
            f'<div class="result-desc">{desc}</div></div>', unsafe_allow_html=True)
        st.markdown("**💡 Recommendations:**")
        for tip in tips:
            st.markdown(f"- {tip}")

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ─── Tabs for Visualizations ───
tab1, tab2, tab3 = st.tabs(["📉 Cluster Visualization", "📊 Data Insights", "🔬 How It Works"])

with tab1:
    viz_choice = st.radio("Choose Cluster:", ["Sleep Behavior", "Academic Performance"], horizontal=True)
    if viz_choice == "Sleep Behavior":
        model_key = "kmeans_sleep" if model_choice == "KMeans" else "gmm_sleep"
        map_key = "sleep_mapping" if model_choice == "KMeans" else "gmm_sleep_mapping"
        labels_map = SLEEP_LABELS
    else:
        model_key = "kmeans_academic" if model_choice == "KMeans" else "gmm_academic"
        map_key = "academic_mapping" if model_choice == "KMeans" else "gmm_academic_mapping"
        labels_map = ACADEMIC_LABELS

    data_clean = df[FEATURES].dropna()
    data_scaled = models["scaler"].transform(data_clean)
    clusters = models[model_key].predict(data_scaled)
    mapping = models[map_key]
    mapped = [labels_map.get(mapping.get(c, c), str(c)) for c in clusters]

    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(data_scaled)

    user_feats = [user_input["Study_Hours"], user_input["Screen_Time"],
                  user_input["Caffeine_Intake"], user_input["Physical_Activity"] / 2 / 60,
                  user_input["Sleep_Duration"], user_input["Sleep_Quality"]]
    user_pca = pca.transform(models["scaler"].transform([user_feats]))

    viz_df = pd.DataFrame(pca_data, columns=["PCA1", "PCA2"])
    viz_df["Cluster"] = mapped

    fig = px.scatter(viz_df, x="PCA1", y="PCA2", color="Cluster",
                     color_discrete_sequence=px.colors.qualitative.Pastel,
                     opacity=0.75, title=f"{viz_choice} Clusters ({model_choice})")
    fig.add_trace(go.Scatter(x=[user_pca[0, 0]], y=[user_pca[0, 1]],
                             mode="markers", name="⭐ You",
                             marker=dict(size=16, color="#FF6584", symbol="x",
                                         line=dict(width=2, color="white"))))
    fig.update_layout(template="plotly_dark", height=500,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter"))
    st.plotly_chart(fig, width='stretch')

with tab2:
    st.subheader("Feature Distributions")
    feat_sel = st.selectbox("Select feature:", FEATURES)
    fig_hist = px.histogram(df, x=feat_sel, nbins=20, marginal="box",
                            color_discrete_sequence=["#6C63FF"],
                            title=f"Distribution of {feat_sel}")
    fig_hist.update_layout(template="plotly_dark", height=400,
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_hist, width='stretch')

    st.subheader("Correlation Heatmap")
    corr = df[FEATURES].corr()
    fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                         aspect="auto", title="Feature Correlations")
    fig_corr.update_layout(template="plotly_dark", height=450,
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_corr, width='stretch')

    st.subheader("Elbow Method (KMeans)")
    data_scaled = models["scaler"].transform(df[FEATURES].dropna())
    max_k = min(10, len(data_scaled))
    if max_k > 2:
        sse = []
        for k in range(2, max_k):
            km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(data_scaled)
            sse.append(km.inertia_)
        fig_elbow = px.line(x=list(range(2, max_k)), y=sse, markers=True,
                            labels={"x": "Number of Clusters (k)", "y": "SSE"},
                            title="Elbow Method for Optimal k")
        fig_elbow.update_traces(line_color="#E040FB", marker_color="#6C63FF")
        fig_elbow.update_layout(template="plotly_dark", height=400,
                                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_elbow, width='stretch')

with tab3:
    st.markdown("""
    ### 🧠 How It Works

    This application uses **unsupervised machine learning** to categorize students based on their lifestyle patterns.

    **1. Data Collection**
    - Real survey data from 400+ students covering study habits, screen time, caffeine intake, physical activity, and sleep patterns.

    **2. Feature Engineering**
    - Six key features are extracted and standardized using `StandardScaler` for uniform model input.

    **3. Clustering Models**
    - **KMeans**: Partitions students into 3 distinct groups based on distance to cluster centroids.
    - **GMM (Gaussian Mixture Model)**: Uses probabilistic soft-assignment for more nuanced clustering.

    **4. Cluster Interpretation**
    - Sleep clusters → Night Owl · Balanced Sleeper · Oversleeper
    - Academic clusters → Low · Average · High Performer
    - Clusters are ordered by the target feature (Sleep Duration / Study Hours) for consistent labeling.

    **5. Visualization**
    - PCA reduces the 6D feature space to 2D for intuitive scatter plot visualization.
    - Your position (⭐) is overlaid on the cluster map so you can see where you fall.
    """)

# ─── Footer ───
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#616161; font-size:0.85rem; padding:1rem 0;">
    Built by <strong>Subham Kumar</strong> •
    <a href="https://github.com/im-subhamkumar/Sleep_Pattern_Prediction_AI-ML" target="_blank" style="color:#6C63FF;">GitHub Repo</a> •
    Powered by Streamlit & scikit-learn
</div>
""", unsafe_allow_html=True)
