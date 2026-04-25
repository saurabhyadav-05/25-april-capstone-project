import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("retail_sales_data.csv")

print(df.head())

# PREPROCESSING

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

# FEATURES & TARGET

X = df[
    [
        "ProductID",
        "Category",
        "Region",
        "Price",
        "Discount",
        "Holiday",
        "Day",
        "Month",
        "Year",
    ]
]

y = df["UnitsSold"]

# TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# TRAIN MODEL

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

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

    mlflow.sklearn.log_model(
        sk_model=model,
        name="DemandModel",
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_SKOPS,
    )

    print("Model Trained Successfully")
    print("MAE:", mae)
    print("RMSE:", rmse)

# SAVE MODEL

joblib.dump(model, "demand_model.pkl")
joblib.dump(le_product, "product_encoder.pkl")
joblib.dump(le_category, "category_encoder.pkl")
joblib.dump(le_region, "region_encoder.pkl")

# FASTAPI APP

app = FastAPI(
    title="Smart Retail Demand Forecast API",
    description="Predict Product Demand",
    version="1.0"
)

# Redirect home to Swagger UI
@app.get("/", include_in_schema=False)
def home():
    return RedirectResponse("/docs")

# INPUT SCHEMA

class DemandRequest(BaseModel):
    ProductID: str
    Category: str
    Date: str
    Region: str
    Price: float
    Discount: int
    Holiday: int

# API ENDPOINT

@app.post("/predict-demand")
def predict_demand(data: DemandRequest):

    # Convert Input Date
    input_date = pd.to_datetime(data.Date)

    day = input_date.day
    month = input_date.month
    year = input_date.year

    # Encode Values
    product = le_product.transform([data.ProductID])[0]
    category = le_category.transform([data.Category])[0]
    region = le_region.transform([data.Region])[0]

    # Create DataFrame
    input_df = pd.DataFrame([{
        "ProductID": product,
        "Category": category,
        "Region": region,
        "Price": data.Price,
        "Discount": data.Discount,
        "Holiday": data.Holiday,
        "Day": day,
        "Month": month,
        "Year": year
    }])

    # Predict
    result = model.predict(input_df)[0]

    return {
        "ProductID": data.ProductID,
        "Region": data.Region,
        "Predicted_Demand": round(float(result), 2),
        "Status": "Success"
    }