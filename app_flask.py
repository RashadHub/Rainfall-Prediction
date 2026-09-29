"""
Rainfall Prediction Web Application (Flask + HTML/CSS/JS)
Run: python app_flask.py
"""
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("rainfall_model.pkl")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    rainfall_index = (
        float(data["humidity_max"]) * float(data["cloud_cover_mean"]) * float(data["wind_speed_max"])
    ) / (float(data["pressure_min"]) + 1)

    input_df = pd.DataFrame({
        "avg_temp_ksa": [float(data["avg_temp_ksa"])],
        "avg_wind_ksa": [float(data["avg_wind_ksa"])],
        "avg_humidity_ksa": [float(data["avg_humidity_ksa"])],
        "avg_barometer_ksa": [float(data["avg_barometer_ksa"])],
        "avg_visibility_ksa": [float(data["avg_visibility_ksa"])],
        "temp_max": [float(data["temp_max"])],
        "temp_min": [float(data["temp_min"])],
        "precipitation_sum": [float(data["precipitation_sum"])],
        "rain_sum": [float(data["rain_sum"])],
        "wind_speed_max": [float(data["wind_speed_max"])],
        "wind_gusts_max": [float(data["wind_gusts_max"])],
        "humidity_max": [int(data["humidity_max"])],
        "humidity_min": [int(data["humidity_min"])],
        "pressure_max": [float(data["pressure_max"])],
        "pressure_min": [float(data["pressure_min"])],
        "cloud_cover_mean": [int(data["cloud_cover_mean"])],
        "Rainfall_Index": [rainfall_index],
        "location_id": [int(data["location_id"])],
    })

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    return jsonify({
        "prediction": int(prediction),
        "probability_rain": round(float(probability[1]) * 100, 1),
        "probability_no_rain": round(float(probability[0]) * 100, 1),
    })


if __name__ == "__main__":
    app.run(debug=False, port=5000)
