#import streamlit as st
import pandas as pd
import joblib
# Importem aquestes classes perquè el Pipeline les necessita per carregar-se rectament
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# 1. Configuració de la pàgina
st.set_page_config(page_title="Machine Failure Predictor", page_icon="🛠️")

# 2. Carregar el model
# El fitxer 'predictive_maintenance_model.pkl' ha d'estar a la mateixa carpeta
try:
    model = joblib.load('predictive_maintenance_model.pkl')
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

# 3. Interfície de l'aplicació
st.title("🛠️ Predictive Maintenance Dashboard")
st.write("""
This application uses a Machine Learning model to predict machine failures 
before they occur, based on real-time sensor data.
""")

st.sidebar.header("Sensor Input Data")

def user_input_features():
    # He posat els noms exactes que espera el teu Pipeline segons el notebook
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

input_df = user_input_features()

# 4. Mostrar dades introduïdes
st.subheader("Current Sensor Readings")
st.write(input_df)

# 5. Predicció
if st.button("Analyze Machine Status"):
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[0][1]
    
    st.subheader("Diagnostic Results")
    
    if prediction[0] == 1:
        st.error(f"⚠️ **CRITICAL:** High risk of failure detected!")
        st.write(f"Failure Probability: **{probability:.2%}**")
        st.info("Recommended Action: Schedule immediate maintenance check.")
    else:
        st.success(f"✅ **NORMAL:** Machine is operating within safe parameters.")
        st.write(f"Probability of failure: **{probability:.2%}**")
        st.info("Recommended Action: Continue standard monitoring.")

st.markdown("---")
st.caption("Model trained on AI4I 2020 Predictive Maintenance Dataset.")
