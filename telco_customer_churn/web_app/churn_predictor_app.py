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
columns_path = Path.cwd() / 'telco_customer_churn' / 'serialization' / 'columns.json'
try:
    with open(columns_path, "r") as f:
        columns_dict = json.loads(f.read())
        column_names = columns_dict['columns']
except Exception as e:
    print("Error while loading columns: ", e)


from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory='templates')

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

@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post('/predict')
async def predict_churn(customer_input: CustomerInput):
    input_data = customer_input.model_dump()
    input_data['TotalCharges']=input_data['tenure']*input_data['MonthlyCharges']
    input_data = pd.DataFrame([input_data])
    X = transformer.transform(input_data)
    X = pd.DataFrame(X, columns=column_names)
    print(X)
    prediction = model.predict(X)
    # return {"churn_prediction": bool(prediction[0])}
    return X.to_dict()


#   "Gender": "Male",
#   "SeniorCitizen": "Yes",
#   "Partner": "Yes",
#   "Dependents": "Yes",
#   "MultipleLines": "Yes",
#   "InternetService": "DSL",
#   "OnlineSecurity": "Yes",
#   "OnlineBackup": "Yes",
#   "DeviceProtection": "Yes",
#   "TechSupport": "Yes",
#   "StreamingTV": "Yes",
#   "StreamingMovies": "Yes",
#   "PaperlessBilling": "Yes",
#   "PaymentMethod": "Mailed check",
#   "Contract": "One year",
#   "Tenure": 2,
#   "MonthlyCharges": 30.00