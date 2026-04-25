import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn

from fastapi import FastAPI
from pydantic import BaseModel

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder


# ✅ FIX: Use raw string (r"...") to avoid backslash escape issues
df = pd.read_csv(r"C:\Users\Asus\Downloads\retail_sales_data.csv")

print(df.head())

# STEP 3: PREPROCESSING

df["Date"] = pd.to_datetime(df["Date"])
df["Day"] = df["Date"].dt.day
df["Month"] = df["Date"].dt.month
df["Year"] = df["Date"].dt.year

le_product = LabelEncoder()
le_category = LabelEncoder()
le_region = LabelEncoder()

df["ProductID"] = le_product.fit_transform(df["ProductID"])
df["Category"] = le_category.fit_transform(df["Category"])
df["Region"] = le_region.fit_transform(df["Region"])

# STEP 4: FEATURES & TARGET

X = df[[
    "ProductID",
    "Category",
    "Region",
    "Price",
    "Discount",
    "Holiday",
    "Day",
    "Month",
    "Year"
]]

y = df["UnitsSold"]

# STEP 5: TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# STEP 6: TRAIN MODEL

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

# STEP 7: MODEL VERSIONING USING MLFLOW

mlflow.set_experiment("Retail_Demand_Forecasting")

with mlflow.start_run():

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    mlflow.log_param("Model", "RandomForest")
    mlflow.log_param("Trees", 200)

    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("RMSE", rmse)

    mlflow.sklearn.log_model(model, "DemandModel")

    print("Model Trained Successfully")
    print("MAE:", mae)
    print("RMSE:", rmse)

# Save Model
joblib.dump(model, "demand_model.pkl")

# Save Encoders
joblib.dump(le_product, "product_encoder.pkl")
joblib.dump(le_category, "category_encoder.pkl")
joblib.dump(le_region, "region_encoder.pkl")

# STEP 8: DATA DRIFT DETECTION

old_avg_price = df["Price"].mean()

# ✅ FIX: Same raw string fix applied here too
new_data = pd.read_csv(r"C:\Users\Asus\Downloads\retail_sales_data.csv")
new_avg_price = new_data["Price"].mean()

if abs(old_avg_price - new_avg_price) > 5000:
    print("⚠️ Data Drift Detected")
else:
    print("✅ No Drift")

# STEP 9: FASTAPI DEPLOYMENT


app = FastAPI(title="Retail Demand Forecast API")

loaded_model = joblib.load("demand_model.pkl")
le_product = joblib.load("product_encoder.pkl")
le_category = joblib.load("category_encoder.pkl")
le_region = joblib.load("region_encoder.pkl")

class DemandRequest(BaseModel):
    ProductID: str
    Category: str
    Region: str
    Price: float
    Discount: int
    Holiday: int
    Day: int
    Month: int
    Year: int

@app.get("/")
def home():
    return {"message": "Retail Demand Forecast API Running"}

@app.post("/predict-demand")
def predict(data: DemandRequest):

    product = le_product.transform([data.ProductID])[0]
    category = le_category.transform([data.Category])[0]
    region = le_region.transform([data.Region])[0]

    input_data = pd.DataFrame([{
        "ProductID": product,
        "Category": category,
        "Region": region,
        "Price": data.Price,
        "Discount": data.Discount,
        "Holiday": data.Holiday,
        "Day": data.Day,
        "Month": data.Month,
        "Year": data.Year
    }])

    prediction = loaded_model.predict(input_data)[0]

    return {
        "Predicted_Demand": round(float(prediction), 2)
    }