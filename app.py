"""
Streamlit demo for the predictive maintenance model.

It trains a random forest on data/ai4i2020.csv the first time it loads (cached), using the
same choices as the notebook: drop the leakage columns, add a power feature and a
temperature gap feature, and use balanced class weights for the 3.4% failure rate. Then it
lets you move the sensor readings and see the failure probability. Full write-up in
analysis.ipynb.

Run locally:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Predictive Maintenance", page_icon="gear", layout="centered")

TYPE_MAP = {"L (low)": 0, "M (medium)": 1, "H (high)": 2}


@st.cache_resource
def train_model():
    df = pd.read_csv("data/ai4i2020.csv")
    df = df.rename(columns={
        "Air temperature [K]": "air_temp",
        "Process temperature [K]": "process_temp",
        "Rotational speed [rpm]": "rpm",
        "Torque [Nm]": "torque",
        "Tool wear [min]": "tool_wear",
        "Machine failure": "failure",
    })
    df["type_code"] = df["Type"].map({"L": 0, "M": 1, "H": 2})
    # Engineered features from the physics.
    df["power"] = df["torque"] * df["rpm"] * 2 * np.pi / 60.0
    df["temp_gap"] = df["process_temp"] - df["air_temp"]
    feats = ["type_code", "air_temp", "process_temp", "rpm", "torque",
             "tool_wear", "power", "temp_gap"]
    X, y = df[feats], df["failure"]
    model = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                   random_state=42, n_jobs=-1)
    model.fit(X, y)
    return model, feats


model, feats = train_model()

st.title("Predictive maintenance, machine failure risk")
st.caption(
    "Flagging that a machine is about to fail from live sensor readings. Trained on the "
    "AI4I 2020 dataset, where only 3.4% of runs fail, so the model is tuned to catch "
    "failures rather than to be accurate. See analysis.ipynb for the reasoning."
)

col1, col2 = st.columns(2)
with col1:
    mtype = st.selectbox("Product quality type", list(TYPE_MAP.keys()), index=1)
    air_temp = st.slider("Air temperature (K)", 295.0, 305.0, 300.0, 0.1)
    process_temp = st.slider("Process temperature (K)", 305.0, 314.0, 310.0, 0.1)
with col2:
    rpm = st.slider("Rotational speed (rpm)", 1160, 2890, 1540, 1)
    torque = st.slider("Torque (Nm)", 3.5, 77.0, 40.0, 0.1)
    tool_wear = st.slider("Tool wear (min)", 0, 255, 100, 1)

power = torque * rpm * 2 * np.pi / 60.0
temp_gap = process_temp - air_temp
X = pd.DataFrame([{
    "type_code": TYPE_MAP[mtype], "air_temp": air_temp, "process_temp": process_temp,
    "rpm": rpm, "torque": torque, "tool_wear": tool_wear,
    "power": power, "temp_gap": temp_gap,
}])[feats]

prob = float(model.predict_proba(X)[:, 1][0])
st.metric("Estimated failure probability", f"{prob:.1%}")
if prob >= 0.3:
    st.error("Elevated risk. In a real setup this is where I would raise a maintenance flag.")
else:
    st.success("Low risk on these readings.")
st.caption(
    f"Mechanical power at these settings is about {power/1000:.1f} kW, and the process to "
    f"air temperature gap is {temp_gap:.1f} K. Both were among the most useful features."
)
