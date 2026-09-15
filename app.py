import os
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import joblib

# Set Page Config
st.set_page_config(
    page_title="Delivery Time Prediction",
    page_icon="🚚",
    layout="wide"
)

# Handle both inner and outer execution directory
BASE_DIR = Path(__file__).resolve().parent
if not (BASE_DIR / "delivery_time_model.pkl").exists() and (BASE_DIR / "Delivery_Time_Prediction_Sklearn" / "delivery_time_model.pkl").exists():
    BASE_DIR = BASE_DIR / "Delivery_Time_Prediction_Sklearn"

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
    }
    .result-banner {
        background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin-top: 1rem;
        box-shadow: 0 10px 15px -3px rgba(245, 158, 11, 0.3);
    }
    .result-number {
        font-size: 2.3rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🚚 Delivery Time Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predicts order arrival ETA based on route distance, kitchen prep time, weather, and real-time traffic using <b>RandomForestRegressor Pipeline</b>.</div>', unsafe_allow_html=True)

model_path = BASE_DIR / "delivery_time_model.pkl"
csv_path = BASE_DIR / "data" / "delivery_data.csv"
chart_path = BASE_DIR / "feature_importance.png"

if not model_path.exists():
    st.error(f"Model file not found at `{model_path}`! Please run 'python train_model.py' first.")
    st.stop()

@st.cache_resource
def get_model():
    return joblib.load(str(model_path))

model = get_model()

tab1, tab2, tab3 = st.tabs(["🔮 Live ETA Predictor", "📈 Feature Importance & Insights", "📋 Delivery Dataset"])

with tab1:
    col_input, col_result = st.columns([1.1, 0.9])
    
    with col_input:
        st.subheader("Order & Route Details")
        with st.form("delivery_form"):
            distance = st.slider("Delivery Distance (km)", 0.5, 35.0, 5.5, step=0.5)
            prep_time = st.slider("Kitchen / Prep Time (mins)", 5.0, 60.0, 15.0, step=1.0)
            traffic = st.selectbox("Traffic Condition", ["Low", "Medium", "High"], index=1)
            weather = st.selectbox("Weather Condition", ["Clear", "Rainy", "Cloudy"], index=0)
            vehicle = st.selectbox("Delivery Vehicle", ["Bike", "Scooter", "Car"], index=0)
            
            submit_btn = st.form_submit_button("🚀 Calculate Estimated Delivery Time", use_container_width=True)
            
    with col_result:
        st.subheader("ETA Forecast")
        if submit_btn:
            sample = pd.DataFrame([{
                "distance_km": distance,
                "preparation_time_min": prep_time,
                "traffic_level": traffic,
                "weather": weather,
                "vehicle_type": vehicle
            }])
            
            pred = model.predict(sample)[0]
            
            st.markdown(f"""
            <div class="result-banner">
                <div style="font-size: 0.95rem; opacity: 0.9;">Estimated Delivery Time</div>
                <div class="result-number">⏱️ {pred:.0f} mins</div>
                <div style="font-size: 1.05rem; opacity: 0.95;">≈ {pred:.1f} minutes from order placement</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.success("✅ Delivery ETA successfully computed!")
            
            if pred <= 25:
                speed_tier = "⚡ Super Express Delivery (< 25 mins)"
            elif pred <= 45:
                speed_tier = "🚀 Standard Fast Delivery (25 - 45 mins)"
            else:
                speed_tier = "⏳ Extended Transit Delivery (> 45 mins)"
            st.info(f"**Transit Classification**: {speed_tier}")
            
            with st.expander("🔍 Order Input Payload"):
                st.json(sample.to_dict(orient="records")[0])
        else:
            st.info("👈 Adjust trip parameters and click **'Calculate Estimated Delivery Time'**.")

with tab2:
    st.subheader("Feature Importance")
    if chart_path.exists():
        st.image(str(chart_path), caption="Delivery Model Feature Importance", use_container_width=True)
    else:
        st.info("Feature importance plot will appear after running train_model.py")

with tab3:
    st.subheader("Training Dataset (delivery_data.csv)")
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        st.write(f"Total Delivery Records: **{len(df):,}** | Columns: **{len(df.columns)}**")
        st.dataframe(df.head(50), use_container_width=True)
    else:
        st.warning("Dataset not found.")
