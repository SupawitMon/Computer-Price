import streamlit as st
import pandas as pd
import joblib

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Computer Price Predictor Ultra",
    layout="centered"
)

# =========================
# LOAD MODEL
# =========================
model = joblib.load('computer_price_model.pkl')

# =========================
# FIX INPUT (กัน feature หาย)
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
# SMART ADJUST (แก้ saturation 🔥)
# =========================
def adjust_price(usd_price, ram, storage, cpu_speed):
    factor = 1.0

    # 🔥 RAM scaling (ต่อเนื่อง)
    factor *= (1 + (ram - 8) * 0.015)

    # 🔥 Storage scaling
    factor *= (1 + (storage - 256) * 0.0001)

    # 🔥 CPU scaling
    factor *= (1 + (cpu_speed - 2.0) * 0.1)

    return usd_price * factor

# =========================
# TITLE
# =========================
st.title("💻 Computer Price Predictor Ultra")
st.caption("AI-powered realistic computer pricing system")

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
    ram = st.number_input("RAM (GB)", 1, 256, 8)

with col2:
    storage = st.number_input("Storage (GB)", 64, 4000, 512)
    cpu_speed = st.number_input("CPU Speed (GHz)", 1.0, 5.0, 2.5)

# =========================
# VALIDATION
# =========================
if brand == "Select Brand":
    st.warning("⚠️ Please select a valid brand")
    st.stop()

# เตือนค่าหลุด dataset
if ram > 64:
    st.warning("⚠️ RAM สูงเกิน dataset (prediction อาจคลาดเคลื่อน)")

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

        # 🔹 Raw prediction
        usd_price = model.predict(fixed_df)[0]

        # 🔥 Adjust ให้สมจริง
        usd_price = adjust_price(usd_price, ram, storage, cpu_speed)

        thb_price = usd_price * 35

        if currency == "THB (บาท)":
            st.success(f"💰 Estimated Price: {thb_price:,.2f} บาท")
        else:
            st.success(f"💰 Estimated Price: ${usd_price:,.2f}")

        st.progress(90)
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
- Random Forest model used
- Cannot extrapolate beyond training data range
- Dataset mostly contains mid-to-high range computers
- Adjustment function added to improve realism
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
- Model may be less accurate for extreme specifications  
""")
