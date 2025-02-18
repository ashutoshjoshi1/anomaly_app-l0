from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import joblib
import os
from tensorflow.keras import models
import matplotlib.pyplot as plt
import io
import base64
from tensorflow.python import pywrap_tensorflow
import tensorflow as tf

app = Flask(__name__)

# Load the trained model and scaler
loaded_autoencoder = models.load_model('anomaly_detector_model.h5', 
                                       custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
scaler = joblib.load('scaler.pkl')

def process_txt_file(file):
    lines = file.readlines()
    data = []
    for line in lines:
        line = line.strip()
        if not line.startswith('#') and line:
            row = line.split()
            if len(row) >= 24:
                row = row[:24]
                data.append(row)
    
    columns = [
        "Routine Code", "Timestamp", "Routine Count", "Repetition Count", "Duration", "Integration Time [ms]",
        "Number of Cycles", "Saturation Index", "Filterwheel 1", "Filterwheel 2", "Zenith Angle [deg]", "Zenith Mode",
        "Azimuth Angle [deg]", "Azimuth Mode", "Processing Index", "Target Distance [m]",
        "Electronics Temp [\u00b0C]", "Control Temp [\u00b0C]", "Aux Temp [\u00b0C]", "Head Sensor Temp [\u00b0C]",
        "Head Sensor Humidity [%]", "Head Sensor Pressure [hPa]", "Scale Factor", "Uncertainty Indicator"
    ]

    df = pd.DataFrame(data, columns=columns)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"].str.replace("T", " ").str.replace("Z", ""), errors='coerce')
    df = df.dropna(subset=["Timestamp"])
    df_numeric = df.drop(columns=["Routine Code", "Timestamp"], errors='ignore')
    df_numeric = df_numeric.apply(pd.to_numeric, errors='coerce')
    
    return df, df_numeric

def detect_anomalies(df_numeric):
    df_scaled = scaler.transform(df_numeric)
    reconstructions = loaded_autoencoder.predict(df_scaled)
    reconstruction_errors = np.mean(np.abs(df_scaled - reconstructions), axis=1)
    threshold = np.percentile(reconstruction_errors, 99.9)
    return reconstruction_errors > threshold

def plot_results(df, anomalies):
    normal_data = df[~anomalies]
    anomalous_data = df[anomalies]
    img = io.BytesIO()
    for column in df.columns:
        plt.figure(figsize=(12, 6))
        plt.plot(normal_data['Timestamp'], normal_data[column], label='Normal', alpha=0.5)
        plt.scatter(anomalous_data['Timestamp'], anomalous_data[column], color='red', label='Anomaly', marker='x')
        plt.title(f'{column} - Anomaly Detection')
        plt.xlabel('Timestamp')
        plt.ylabel(column)
        plt.legend()
        plt.grid(True)
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()
    
    return base64.b64encode(img.getvalue()).decode('utf8')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    df, df_numeric = process_txt_file(file)
    anomalies = detect_anomalies(df_numeric)
    plot_url = plot_results(df, anomalies)
    
    return jsonify({'anomalies_detected': int(anomalies.sum()), 'plot_url': plot_url})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
