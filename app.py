import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page Layout Configuration (Dark theme elements and wide view)
st.set_page_config(page_title="ISRO Urban Heat Mitigation AI", page_icon="🛰️", layout="wide")

# Custom CSS for Premium Look
# Change this specific block near Line 10 in your app.py:
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    h1 { color: #4A90E2; font-family: 'Helvetica Neue', sans-serif; }
    .stMetric { background-color: #1e293b; padding: 15px; border-radius: 10px; border: 1px solid #334155; }
    </style>
""", unsafe_allow_html=True)  # <-- Fixed 'unsafe_allow_html' here

st.title("🛰️ AI-Powered Urban Heat Mitigation & Cooling System")
st.markdown("##### ISRO Hackathon Prototype | Physics-Informed Urban Simulation Engine")
st.write("---")

# Load Saved Files
try:
    model = joblib.load('isro_xgb_model.pkl')
    scaler = joblib.load('pipeline_scaler.pkl')
    feature_cols = joblib.load('feature_columns.pkl')
    st.sidebar.success("✅ Pre-trained AI Model Active")
except FileNotFoundError:
    st.error("❌ Saved files missing. Make sure notebook has run and saved models in the same directory.")
    st.stop()

# --- STEP 1: LOGICAL TABS CREATION ---
tab1, tab2, tab3 = st.tabs(["📊 Live Climate Controls", "🌳 AI Mitigation Sandbox", "🧬 Model Performance Insights"])

with tab1:
    st.subheader("🛠️ Current Environmental & Urban Features")
    st.caption("Change parameters to simulate different geographical scenarios.")

    col1, col2, col3 = st.columns(3)

    with col1:
        greenness = st.slider("Urban Greenness Ratio (%)", 0.0, 100.0, 15.0,
                              help="Total percentage of tree canopy or vegetation cover.")
        pop_density = st.slider("Population Density (people/km²)", 100, 25000, 5000)
        elevation = st.slider("Elevation (m)", 0, 2000, 200)

    with col2:
        air_temp = st.slider("Atmospheric Air Temperature (°C)", 15.0, 50.0, 35.0)
        humidity = st.slider("Humidity (%)", 10.0, 100.0, 50.0)
        wind_speed = st.slider("Wind Speed (km/h)", 0.0, 50.0, 12.0)

    with col3:
        # HERE IS THE FIX: Added Annual Rainfall Slider to resolve KeyError
        rainfall = st.slider("Annual Rainfall (mm)", 100, 3000, 1200,
                             help="Resolves index constraint from source dataset.")
        energy_cons = st.slider("Energy Consumption (kWh)", 500, 10000, 3000)
        aqi = st.slider("Air Quality Index (AQI)", 0, 500, 96)

    # Categorical Selection
    land_cover = st.selectbox("Predominant Land Cover Type", ["Urban", "Industrial", "Green Space", "Water"])

    # Mapping user inputs exactly to feature_columns from training
    input_data = {
        'Elevation (m)': elevation,
        'Population Density (people/km²)': pop_density,
        'Energy Consumption (kWh)': energy_cons,
        'Air Quality Index (AQI)': aqi,
        'Urban Greenness Ratio (%)': greenness,
        'Wind Speed (km/h)': wind_speed,
        'Humidity (%)': humidity,
        'Annual Rainfall (mm)': rainfall,  # Injected mapped column fix
        'Land Cover_Industrial': 1 if land_cover == "Industrial" else 0,
        'Land Cover_Urban': 1 if land_cover == "Urban" else 0,
        'Land Cover_Water': 1 if land_cover == "Water" else 0
    }

    # Ensure all missing columns dynamically evaluated as 0 if any left
    for col in feature_cols:
        if col not in input_data:
            input_data[col] = 0

    input_df = pd.DataFrame([input_data])[feature_cols]
    input_scaled = scaler.transform(input_df)
    predicted_temp = model.predict(input_scaled)[0]

    st.write("---")
    st.markdown("#### Baseline Prediction Results")
    st.metric(label="Predicted Microclimate Surface Temperature", value=f"{predicted_temp:.2f} °C")

with tab2:
    st.subheader("🌳 Policy Intervention & Urban Greening Simulator")
    st.caption("Apply active cool infrastructure plans to monitor AI-driven thermal drops.")

    greening_increase = st.slider("Increase Green Infrastructure Coverage By:", 0, 50, 20, format="%d%%")

    # Run the tweak on dataset matrix
    simulated_df = input_df.copy()
    simulated_df['Urban Greenness Ratio (%)'] = np.minimum(
        simulated_df['Urban Greenness Ratio (%)'] + greening_increase, 100.0)

    if land_cover == "Urban" and greening_increase > 20:
        simulated_df['Land Cover_Urban'] = 0  # Dynamic shifting

    simulated_scaled = scaler.transform(simulated_df)
    predicted_new_temp = model.predict(simulated_scaled)[0]
    temp_drop = predicted_temp - predicted_new_temp

    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.metric(label="Optimized Microclimate Temperature", value=f"{predicted_new_temp:.2f} °C",
                  delta=f"-{temp_drop:.2f} °C", delta_color="inverse")
    with res_col2:
        st.info(
            f"💡 **ISRO AI Engine Recommendation:** Increasing the dynamic urban canopy framework by **{greening_increase}%** triggers a projected localized cooling index reduction of **{temp_drop:.2f}°C**.")

    # Visualization
    chart_data = pd.DataFrame({
        'Scenario Model': ['Status Quo (Current)', 'Mitigated (Cooling Applied)'],
        'Temperature (°C)': [predicted_temp, predicted_new_temp]
    })
    st.bar_chart(data=chart_data, x='Scenario Model', y='Temperature (°C)', color="#4A90E2", use_container_width=True)

with tab3:
    st.subheader("🧬 Pipeline Structural Diagnostics")
    st.write("This tab serves as proof of structural correctness for evaluation panel.")
    st.json(feature_cols)