from fastapi import FastAPI, Request
from pydantic import BaseModel
import pickle
import json
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional
import uvicorn

app = FastAPI()

# Import saved preprocessor to transform features
preprocessor_path = Path.cwd() / 'telco_customer_churn' / 'serialization' / 'preprocessor.pickle'
try:
    with open(preprocessor_path, "rb") as f:
        transformer = pickle.load(f)
except Exception as e:
    print("Error while loading transformer: ", e)

# Import saved model
model_path = Path.cwd() / 'telco_customer_churn' / 'serialization' / 'churn_predictor.pickle'
try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    print("Error while loading model: ", e)

# Import feature names
columns_path = Path.cwd() / 'telco_customer_churn' / 'serialization' / 'column_rename.json'
try:
    with open(columns_path, "r") as f:
        column_dict = json.loads(f.read())
        feature_names = column_dict['feature_names']
        column_rename = column_dict['column_rename']
except Exception as e:
    print("Error while loading columns: ", e)


from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory='./telco_customer_churn/web_app/templates')

class CustomerInput(BaseModel):
    
    gender : str
    SeniorCitizen : str
    Partner : str
    Dependents : str
    tenure : int
    MultipleLines : str = 'No'
    InternetService : str
    OnlineSecurity : str = 'No'
    OnlineBackup : str = 'No'
    DeviceProtection : str = 'No'
    TechSupport : str = 'No'
    StreamingTV : str = 'No'
    StreamingMovies : str = 'No'
    Contract : str
    PaperlessBilling : str
    PaymentMethod : str
    MonthlyCharges : float
    TotalCharges : float

@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post('/predict')
async def predict_churn(customer_input: CustomerInput):
    input_data = customer_input.model_dump()
    input_data = pd.DataFrame([input_data])
    X = transformer.transform(input_data)
    X = pd.DataFrame(X, columns=feature_names)
    X.rename(columns=column_rename, inplace=True)
    prediction = model.predict(X)
    return {"churn_prediction": bool(prediction[0])}

# # Test Case
#   "gender": "Male",
#   "SeniorCitizen": "No",
#   "Partner": "No",
#   "Dependents": "No",
#   "tenure": 34,
#   "MultipleLines": "No",
#   "InternetService": "DSL",
#   "OnlineSecurity": "Yes",
#   "OnlineBackup": "No",
#   "DeviceProtection": "Yes",
#   "TechSupport": "No",
#   "StreamingTV": "No",
#   "StreamingMovies": "No",
#   "Contract": "One year",
#   "PaperlessBilling": "No",
#   "PaymentMethod": "Mailed check",
#   "MonthlyCharges": 56.95,
#   "TotalCharges": 1889.5