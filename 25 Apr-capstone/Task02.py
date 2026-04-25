import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.sklearn
import skops.io as sio

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score, classification_report

# FILE PATH

file_name = "machine_sensor_data.csv"

if not os.path.exists(file_name):
    print("File not found")
    exit()

# LOAD DATA

df = pd.read_csv(file_name)

print("\nOriginal Data")
print(df.head())

# PREPROCESSING

if "Timestamp" in df.columns:
    df.drop(columns=["Timestamp"], inplace=True)

if "MachineID" in df.columns:
    df["MachineID"] = df["MachineID"].astype("category").cat.codes

df = df.apply(pd.to_numeric, errors="coerce")
df.fillna(df.mean(numeric_only=True), inplace=True)

print("\nCleaned Data")
print(df.head())

# FEATURES / TARGET

X = df.drop("Failure", axis=1)
y = df["Failure"]

# TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# TRAIN MODEL + MLFLOW

mlflow.set_experiment("Predictive-Maintenance")

with mlflow.start_run():

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print("\nAccuracy:", round(acc, 4))
    print("\nClassification Report")
    print(classification_report(y_test, y_pred))

    mlflow.log_metric("accuracy", acc)
    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("n_estimators", 100)

    mlflow.sklearn.log_model(
        sk_model=model,
        name="PredictiveMaintenanceModel",
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_SKOPS
    )

# ANOMALY DETECTION (AIOPS)

anomaly_model = IsolationForest(
    contamination=0.05,
    random_state=42
)

anomaly_model.fit(X)

df["Anomaly"] = anomaly_model.predict(X)

anomalies = df[df["Anomaly"] == -1]

print("\nAnomalies Detected:", len(anomalies))

# SAVE MODEL

sio.dump(model, "predictive_model.skops")

print("\nModel saved as predictive_model.skops")

# LOAD MODEL

loaded_model = sio.load("predictive_model.skops")

# SAMPLE PREDICTION

sample = X_test.iloc[0:1]

prediction = loaded_model.predict(sample)

print("\nSample Prediction")
print("0 = No Failure")
print("1 = Failure")
print("Predicted:", int(prediction[0]))

# FASTAPI APP

app = FastAPI(
    title="Predictive Maintenance API",
    description="Machine Failure Prediction System",
    version="1.0"
)

# Redirect root to Swagger UI
@app.get("/", include_in_schema=False)
def home():
    return RedirectResponse("/docs")

# INPUT MODEL

class SensorInput(BaseModel):
    temperature: float
    vibration: float
    pressure: float

# FAILURE PREDICTION API

@app.post("/predict-failure")
def predict_failure(data: SensorInput):

    # Create full input row based on dataset structure
    # Extra columns defaulted as 0 if needed
    row = {}

    for col in X.columns:
        if col.lower() == "temperature":
            row[col] = data.temperature
        elif col.lower() == "vibration":
            row[col] = data.vibration
        elif col.lower() == "pressure":
            row[col] = data.pressure
        else:
            row[col] = 0

    input_df = pd.DataFrame([row])

    # Prediction
    pred = loaded_model.predict(input_df)[0]

    # Probability
    prob = loaded_model.predict_proba(input_df)[0][1]

    # Alert Logic
    alert = "YES - Send Email/SMS" if prob > 0.70 else "NO"

    return {
        "Failure_Prediction": int(pred),
        "Failure_Probability": round(float(prob), 4),
        "Alert_Trigger": alert,
        "Status": "Success"
    }

# OPTIONAL DASHBOARD DATA ROUTE

@app.get("/dashboard-summary")
def dashboard():

    total_records = len(df)
    total_anomalies = len(anomalies)

    return {
        "Total_Records": total_records,
        "Anomalies_Detected": total_anomalies,
        "Accuracy": round(float(acc), 4),
        "Ready_For_PowerBI": "Yes"
    }