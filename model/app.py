# Save as app.py and run with: streamlit run app.py

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

# Load model and scaler
xgb_model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")

# Optional: Load dataset for visualization
df = pd.read_csv("synthetic_300_rows_dataset.csv")  # Replace with your file if different

# Page configuration
st.set_page_config(page_title="3D Printing Strength Predictor", layout="wide")

# Custom CSS styling
st.markdown("""
    <style>
        .main { background-color: #f8f9fa; }
        h1 { color: #0066cc; font-weight: bold; }
        .stSlider > div { color: #333333; }
    </style>
""", unsafe_allow_html=True)

# Sidebar navigation
page = st.sidebar.radio("🔍 Select Page", ["📊 Predict Strength", "📈 Visualize Data"])

st.title("🧠 3D Printing Strength Predictor")

# ---------- PAGE 1: PREDICTION ----------
if page == "📊 Predict Strength":
    st.markdown("Adjust the print settings below to predict **Tensile** and **Flexural Strength** of the 3D-printed part.")

    col1, col2 = st.columns(2)
    with col1:
        layer_thickness = st.slider("Layer Thickness (μm)", 50, 400, 100)
        nozzle_temp = st.slider("Nozzle Temperature (°C)", 180, 260, 220)
    with col2:
        infill_density = st.slider("Infill Density (%)", 10, 100, 50)
        material = st.selectbox("Material", [0, 1, 2], format_func=lambda x: ["PLA", "ABS", "Other"][x])

    if st.button("🚀 Predict"):
        features = np.array([[layer_thickness, infill_density, nozzle_temp, material]])
        features_scaled = scaler.transform(features)
        prediction = xgb_model.predict(features_scaled)

        tensile, flexural = prediction[0]

        st.success("✅ Prediction Complete")
        colA, colB = st.columns(2)
        colA.metric("Tensile Strength (MPa)", f"{tensile:.2f}")
        colB.metric("Flexural Strength (MPa)", f"{flexural:.2f}")

        # Radar chart
        st.subheader("📡 Prediction Radar Chart")
        fig = go.Figure(data=go.Scatterpolar(
            r=[tensile, flexural],
            theta=['Tensile Strength', 'Flexural Strength'],
            fill='toself',
            line_color='royalblue'
        ))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, max(tensile, flexural) + 10])))
        st.plotly_chart(fig, use_container_width=True)

# ---------- PAGE 2: VISUALIZATION ----------
elif page == "📈 Visualize Data":
    st.markdown("Explore relationships in the dataset used to train the model.")

    st.subheader("💡 Strength vs. Parameters")
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.scatter(df, x="Layer Thickness (μm)", y="Tensile Strength (MPa)", color=df["Material"].map({0: "PLA", 1: "ABS", 2: "Other"}), title="Layer Thickness vs. Tensile Strength")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.scatter(df, x="Infill Density (%)", y="Flexural Strength (MPa)", color=df["Material"].map({0: "PLA", 1: "ABS", 2: "Other"}), title="Infill Density vs. Flexural Strength")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📊 Correlation Heatmap")
    corr = df.drop(columns=["Material"]).corr()
    fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="blues", title="Correlation Matrix")
    st.plotly_chart(fig_corr, use_container_width=True)

# ---------- Footer ----------
st.markdown("---")
st.caption("🔧 Built with Streamlit · 📦 Model: XGBoost · 🔬 Dataset: Synthetic 3D Printing Data")
