import uvicorn
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal

# Application startup message.
print("--- API: App is starting up... ---")

# Initialize the FastAPI application with metadata.
app = FastAPI(
    title="Customer Churn Prediction API",
    description="An API to predict customer churn using a machine learning model.",
    version="1.0.0"
)

# --- Model and Preprocessor Loading ---
try:
    # Load the preprocessing pipeline (e.g., ColumnTransformer, StandardScaler, etc.).
    preprocessor = joblib.load("preprocessor_pipeline.joblib")
    print("--- API: 'preprocessor_pipeline.joblib' loaded successfully. ---")
    # Load the trained machine learning model.
    model = joblib.load("churn_model.joblib")
    print("--- API: 'churn_model.joblib' loaded successfully. ---")

except FileNotFoundError:
    # Graceful handling for missing model/preprocessor files.
    print("---! API: Model or preprocessor file not found! !---")
    print("---! Make sure the .joblib files are in the same folder. !---")
    preprocessor = None
    model = None

# --- Data Validation and Input Schema ---
class CustomerInput(BaseModel):
    # Pydantic model for request body validation.
    # Literal enforces strict categorical values.
    gender : Literal['Male', 'Female']
    SeniorCitizen: Literal[0,1]
    Partner: Literal['Yes', 'No']
    Dependents: Literal['Yes', 'No']
    # Field enforces data constraints (e.g., non-negative tenure).
    tenure: int = Field(..., ge=0, description="Months customer has stayed")
    PhoneService: Literal['Yes', 'No']
    MultipleLines: Literal['Yes', 'No', 'No phone service']
    InternetService: Literal['DSL', 'Fiber optic', 'No']
    OnlineSecurity: Literal['Yes', 'No', 'No internet service']
    OnlineBackup: Literal['Yes', 'No', 'No internet service']
    DeviceProtection: Literal['Yes', 'No', 'No internet service']
    TechSupport: Literal['Yes', 'No', 'No internet service']
    StreamingTV: Literal['Yes', 'No', 'No internet service']
    StreamingMovies: Literal['Yes', 'No', 'No internet service']
    Contract: Literal['Month-to-month', 'One year', 'Two year']
    PaperlessBilling: Literal['Yes', 'No']
    PaymentMethod: Literal['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)']
    MonthlyCharges: float = Field(..., ge=0) # Enforce non-negative charges.

    # Nested class to provide an example for API documentation (Swagger/Redoc).
    class Config:
        schema_extra = {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85
            }
        }

# --- Prediction Endpoint ---
@app.post("/predict/")
async def predict_churn(customer_data: CustomerInput):
    # Check if resources are available before processing.
    if not model or not preprocessor:
        return {"error": "Model or Preprocessor not loaded. Check server logs."}
    
    # Convert validated Pydantic model to a Pandas DataFrame for processing.
    input_df = pd.DataFrame([customer_data.model_dump()])
    
    # Feature Engineering: Recreating TotalCharges and other model features.
    input_df['total_charges_cleaned'] = input_df['tenure'] * input_df['MonthlyCharges']

    services = [
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
        'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    
    # Calculate ServiceCount feature by converting 'Yes'/'No'/'No internet service' to 1/0/0.
    df_services = input_df[services].replace({'Yes': 1, 'No': 0, 'No internet service': 0})
    input_df['ServiceCount'] = df_services.sum(axis=1)
    
    # Calculate MonthlyCharges_per_Service feature.
    input_df['Monthly_Charges_per_Service'] = input_df['MonthlyCharges'] / (input_df['ServiceCount'] + 1)
    
    print(f"--- API: Received data and engineered features. ---")

    # Transform the raw data using the loaded preprocessor pipeline.
    print(f"--- API: Transforming data... ---")
    processed_features = preprocessor.transform(input_df)

    # Perform the prediction using the loaded model.
    print(f"--- API: Getting prediction... ---")
    # predict_proba returns the probability for each class [P(No Churn), P(Churn)].
    churn_probability_scores = model.predict_proba(processed_features)

    # Extract the probability of churn (usually the second column, index 1).
    churn_probability = churn_probability_scores[0][1]

    print(f"--- API: Prediction complete. Prob: {churn_probability:.4f} ---")
    
    # Return the prediction result as JSON.
    return {
        "churn_probability": round(churn_probability, 4),
        # Determine the final churn status based on a 0.5 threshold.
        "is_churner": bool(churn_probability > 0.5)
    }

# --- Root Endpoint for Health Check / Info ---
@app.get("/")
async def root():
    # Simple message to confirm the API is running.
    return {"message": "Welcome to the Churn Prediction API! Go to /docs to see the magic."}

# --- Application Startup ---
if __name__ == "__main__":
    # Start the uvicorn server for local development.
    uvicorn.run(app, host="127.0.0.1", port=8000)

from mangum import Mangum

handler = Mangum(app)