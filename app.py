import streamlit as st
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# 1. Page Configuration
st.set_page_config(page_title="Maintenance Cost Optimizer")

@st.cache_resource
def load_model():
    return joblib.load('predictive_maintenance_model.pkl')

model = load_model()

# 2. Sidebar - Inputs
st.sidebar.header("Sensor Readings")
m_type = st.sidebar.selectbox("Machine Type", ["L", "M", "H"])
air_temp = st.sidebar.number_input("Air temperature [K]", value=300.0)
proc_temp = st.sidebar.number_input("Process temperature [K]", value=310.0)
rpm = st.sidebar.number_input("Rotational speed [rpm]", value=1500.0)
torque = st.sidebar.number_input("Torque [Nm]", value=40.0)
tool_wear = st.sidebar.number_input("Tool wear [min]", value=0.0)

# 3. Sidebar - Cost Settings (Formal labels)
st.sidebar.header("Cost Parameters (EUR)")
cost_failure = st.sidebar.number_input("Cost of Unplanned Failure", value=5000)
cost_maintenance = st.sidebar.number_input("Cost of Preventive Check", value=500)

# 4. Main Panel
st.title("💰 Predictive Maintenance & Economic Impact")

input_data = pd.DataFrame([{
    'Type': m_type, 'Air temperature [K]': air_temp, 'Process temperature [K]': proc_temp,
    'Rotational speed [rpm]': rpm, 'Torque [Nm]': torque, 'Tool wear [min]': tool_wear
}])

if st.button("Run Economic Analysis"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Technical Diagnostic")
        if prediction == 1:
            st.error("Status: High Risk")
        else:
            st.success("Status: Operational")
        st.write(f"Failure Probability: **{probability:.2%}**")

    with col2:
        st.subheader("Economic Estimate")
        # Lògica: Si el model prediu fallada, l'estalvi és el cost de l'avaria menys el manteniment
        if prediction == 1:
            savings = cost_failure - cost_maintenance
            st.metric("Potential Savings", f"{savings}€", delta="Positive Impact")
            st.write("By intervening now, you avoid a full breakdown cost.")
        else:
            st.metric("Risk-Adjusted Cost", "0€", delta="No Action Needed")
            st.write("The machine is within safe parameters.")

st.markdown("---")
st.caption("Valentí Secanell - Msc Candidate on industrial engineering")
