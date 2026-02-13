import streamlit as st
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# 1. AIXÒ SEMPRE HA D'ANAR PRIMER (just després dels imports)
st.set_page_config(page_title="Machine Failure Predictor", page_icon="🛠️")

# 2. Funció per carregar el model amb memòria cau
@st.cache_resource
def load_model():
    # Assegura't que el nom del fitxer coincideix exactament amb el que tens a GitHub
    return joblib.load('predictive_maintenance_model.pkl')

# 3. Intentem carregar el model
try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# 4. Disseny de la interfície
st.title("🛠️ Predictive Maintenance Dashboard")
st.write("Enter the sensor data to analyze the machine's health status.")

st.sidebar.header("Input Sensor Readings")

def get_user_inputs():
    m_type = st.sidebar.selectbox("Machine Quality Type", ["L", "M", "H"])
    air_temp = st.sidebar.number_input("Air temperature [K]", value=300.0)
    proc_temp = st.sidebar.number_input("Process temperature [K]", value=310.0)
    rpm = st.sidebar.number_input("Rotational speed [rpm]", value=1500.0)
    torque = st.sidebar.number_input("Torque [Nm]", value=40.0)
    tool_wear = st.sidebar.number_input("Tool wear [min]", value=0.0)
    
    data = {
        'Type': m_type,
        'Air temperature [K]': air_temp,
        'Process temperature [K]': proc_temp,
        'Rotational speed [rpm]': rpm,
        'Torque [Nm]': torque,
        'Tool wear [min]': tool_wear
    }
    return pd.DataFrame([data])

input_df = get_user_inputs()

# Mostrar les dades actuals
st.subheader("Current Machine Parameters")
st.write(input_df)

# 5. Botó d'execució
if st.button("Run Diagnostic"):
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[0][1]
    
    st.subheader("Final Diagnostic")
    
    if prediction[0] == 1:
        st.error(f"⚠️ **CRITICAL:** High risk of failure detected!")
        st.write(f"Failure Probability: **{probability:.2%}**")
        st.warning("Action: Inspect the machine immediately.")
    else:
        st.success(f"✅ **NORMAL:** Machine is operating safely.")
        st.write(f"Probability of failure: **{probability:.2%}**")
        st.info("Action: No immediate intervention required.")

st.markdown("---")
st.caption("Valentí Secanell Predictive Maintenance System")
