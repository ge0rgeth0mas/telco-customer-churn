from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import json
from pathlib import Path
import pandas as pd
import numpy as np

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

@app.get('/')
def index():
    return {'data':{'name':'George'}}

@app.get('/about')
def about():
    return {'data':'about page'}