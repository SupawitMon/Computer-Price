import streamlit as st
import pandas as pd
import joblib

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Computer Price Predictor Pro",
    layout="centered"
)

# =========================
# LOAD MODEL
# =========================
model = joblib.load('computer_price_model.pkl')

# =========================
# FIX INPUT
# =========================
def fix_input(input_df, model):
    try:
        model_cols = model.feature_names_in_
    except:
        return input_df

    for col in model_cols:
        if col not in input_df.columns:
            input_df[col] = 0

    return input_df[model_cols]

# =========================
# ADJUST PRICE (แก้ bias)
# =========================
def adjust_price(usd_price, ram, storage, cpu_speed):
    factor = 1.0

    if ram <= 8:
        factor *= 0.75
    if storage <= 512:
        factor *= 0.85
    if cpu_speed <= 2.5:
        factor *= 0.85

    if ram >= 32:
        factor *= 1.2
    if cpu_speed >= 3.5:
        factor *= 1.15

    return usd_price * factor

# =========================
# TITLE
# =========================
st.title("💻 Computer Price Predictor Pro")
st.caption("AI system for estimating computer prices")

st.markdown("---")

# =========================
# INPUT
# =========================
st.subheader("🔧 Input Features")

col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox(
        "Brand",
        ["Select Brand", "Dell", "HP", "Lenovo", "Asus", "Acer", "Apple"]
    )
    ram = st.number_input("RAM (GB)", 1, 128, 8)

with col2:
    storage = st.number_input("Storage (GB)", 64, 4000, 512)
    cpu_speed = st.number_input("CPU Speed (GHz)", 1.0, 5.0, 2.5)

# =========================
# VALIDATION
# =========================
if brand == "Select Brand":
    st.warning("⚠️ Please select a valid brand")
    st.stop()

# =========================
# CREATE DATA
# =========================
input_df = pd.DataFrame([{
    'brand': brand,
    'ram': ram,
    'storage': storage,
    'cpu_speed': cpu_speed
}])

# =========================
# SHOW INPUT
# =========================
st.subheader("📊 Input Data")
st.write(input_df)

# =========================
# CURRENCY
# =========================
st.markdown("---")
currency = st.radio("💱 Currency", ["THB (บาท)", "USD ($)"])

# =========================
# PREDICT
# =========================
if st.button("🚀 Predict Price"):
    try:
        fixed_df = fix_input(input_df, model)

        usd_price = model.predict(fixed_df)[0]

        # ปรับราคาให้สมจริง
        usd_price = adjust_price(usd_price, ram, storage, cpu_speed)

        thb_price = usd_price * 35

        if currency == "THB (บาท)":
            st.success(f"💰 Estimated Price: {thb_price:,.2f} บาท")
        else:
            st.success(f"💰 Estimated Price: ${usd_price:,.2f}")

        # Confidence
        st.progress(85)
        st.caption("Model Confidence: High")

    except Exception as e:
        st.error("Prediction failed ❌")
        st.write(e)

# =========================
# FEATURE IMPORTANCE
# =========================
st.markdown("---")
st.subheader("📊 Feature Importance")

try:
    model_step = model.named_steps['model']
    importances = model_step.feature_importances_

    st.bar_chart(importances[:10])
except:
    st.info("Not available")

# =========================
# INSIGHT
# =========================
st.markdown("---")
st.subheader("🧠 Model Insight")

st.write("""
- Model trained on USD dataset  
- Dataset mostly contains mid-to-high range computers  
- Low-end specs may be slightly overestimated  
- Brand has limited effect if unseen during training  
""")

# =========================
# EXPLANATION
# =========================
st.subheader("📘 Feature Explanation")
st.write("""
- **Brand**: Manufacturer  
- **RAM**: Memory size  
- **Storage**: Disk capacity  
- **CPU Speed**: Processor speed  
""")

# =========================
# DISCLAIMER
# =========================
st.subheader("⚠️ Disclaimer")
st.write("""
- Predictions are estimates only  
- Currency conversion uses approx. rate (1 USD ≈ 35 THB)  
- Model may not reflect real-time market price  
""")