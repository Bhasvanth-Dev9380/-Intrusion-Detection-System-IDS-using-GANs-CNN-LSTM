import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# ✅ Load Trained CNN-LSTM Model
MODEL_PATH = "../models/cnn_lstm_model.h5"
model = load_model(MODEL_PATH)

# ✅ Load Dataset (For Simulation)
DATASET_PATH = "../datasets/NLS_KDD_Original.csv"
data = pd.read_csv(DATASET_PATH)

# ✅ Data Preprocessing
st.sidebar.subheader("📊 Dataset Overview")
st.sidebar.write(f"Loaded dataset with **{data.shape[0]}** rows and **{data.shape[1]}** columns.")

# Remove non-numeric columns
categorical_columns = ['Protocol Type', 'Flag', 'Service', 'Class']
data = data.drop(columns=categorical_columns, errors='ignore')

# Normalize data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Reshape for CNN-LSTM
X_data = data_scaled.reshape(-1, data_scaled.shape[1], 1)

# ✅ Streamlit UI
st.markdown("<h1 style='text-align: center; color: #2E8B57;'>🔍 Intrusion Detection System (IDS) Simulation</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>📡 Real-Time Network Traffic Analysis</h3>", unsafe_allow_html=True)
st.markdown("This IDS model uses a **CNN-LSTM** network to classify network traffic in real-time.")

# Sidebar options
st.sidebar.subheader("⚙️ Simulation Settings")
num_samples = st.sidebar.slider("📥 Number of Packets to Simulate:", 1, 1000, 100)
simulation_speed = st.sidebar.slider("⏳ Simulation Speed (Seconds per Packet)", 0.1, 5.0, 1.0)

# ✅ Live Dataframe to Show Logs
st.markdown("<h3>📡 Intrusion Detection Log (Last 10 Packets)</h3>", unsafe_allow_html=True)
log_placeholder = st.empty()

# ✅ Confidence Threshold
CONFIDENCE_THRESHOLD = 0.70  # 🔥 If confidence < 70%, mark as an attack

# ✅ Simulate Live Network Traffic
if st.button("🚀 Start Live Simulation"):
    detection_logs = []

    for i in range(num_samples):
        # Simulate incoming packet
        incoming_packet = X_data[i].reshape(1, X_data.shape[1], 1)
        prediction = model.predict(incoming_packet)

        predicted_label = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction)  # Get confidence score

        # ✅ Override Classification if Confidence is Low
        if confidence < CONFIDENCE_THRESHOLD:
            predicted_label = 1  # Force it to "Attack"
            classification = "🚨 Forced Attack (Low Confidence)"
            color = "#FF6347"  # Red
        else:
            classification = "✅ Normal" if predicted_label == 0 else "🚨 Attack**"
            color = "#32CD32" if predicted_label == 0 else "#FF6347"

        # ✅ Extract Packet Feature Values for Display
        packet_features = data.iloc[i].to_frame().T  # Convert row to DataFrame (Table)

        # ✅ Format log entry
        log_entry = {
            "Packet ID": f"<span style='color: {color};'><b>{i + 1}</b></span>",
            "Prediction": f"<span style='color: {color};'><b>{classification}</b></span>",
            "Confidence": f"<span style='color: {color};'><b>{confidence * 100:.2f}%</b></span>",
        }

        # ✅ Append new log and show last 10 packets
        detection_logs.append(log_entry)
        detection_logs = detection_logs[-10:]  # Keep only last 10 logs

        # ✅ Display Updated Logs in Streamlit
        log_df = pd.DataFrame(detection_logs)
        log_placeholder.write(log_df.to_html(escape=False), unsafe_allow_html=True)

        # ✅ Display Packet Data **Below Each Packet ID**
        st.markdown(f"<h4 style='color: #4682B4;'>📦 Packet Data (ID: {i+1})</h4>", unsafe_allow_html=True)
        st.dataframe(packet_features)  # Show feature values in a row table

        # ✅ Divider for Better UI
        st.divider()

        # ✅ Delay for Real-Time Effect
        time.sleep(simulation_speed)

    st.success("✅ **Live IDS Simulation Completed!**")

# ✅ Show Confusion Matrix
st.markdown("<h3>📊 Final Confusion Matrix</h3>", unsafe_allow_html=True)
from sklearn.metrics import confusion_matrix

true_labels = np.zeros(num_samples)  # Assuming dataset is all normal (for testing)
predicted_labels = []

for i in range(num_samples):
    incoming_packet = X_data[i].reshape(1, X_data.shape[1], 1)
    prediction = model.predict(incoming_packet)
    confidence = np.max(prediction)

    if confidence < CONFIDENCE_THRESHOLD:
        predicted_label = 1  # Forced Attack
    else:
        predicted_label = np.argmax(prediction, axis=1)[0]

    predicted_labels.append(predicted_label)

cm = confusion_matrix(true_labels, predicted_labels)

fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(fig)

st.sidebar.info("✅ Click 'Start Live Simulation' to see real-time packet classification.")
