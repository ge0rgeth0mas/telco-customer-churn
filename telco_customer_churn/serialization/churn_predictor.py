# Import Basic Libraries
import pandas as pd
import numpy as np
from pathlib import Path
import pickle, json

# Import ML libraries
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression

# Get data
path = Path.cwd() / 'telco_customer_churn' / 'resources' / 'Telco-Customer-Churn.csv' # To ensure path works in both Windows and Mac
data = pd.read_csv(path)

# Data Cleaning
data['SeniorCitizen']=data['SeniorCitizen'].map({0:'No', 1:'Yes'})
data['TotalCharges']=pd.to_numeric(data.TotalCharges, errors='coerce')
data.drop(data[data.TotalCharges.isna()].index, axis=0, inplace=True)
data.drop('customerID', axis=1, inplace=True)
data.reset_index()

# Feature engineering
X = data.drop(['Churn', 'PhoneService'], axis=1)
y = data['Churn'].replace({"Yes":1, "No":0})

internet_addon_services = X.columns[(X == 'No internet service').any()].to_list()
for col in internet_addon_services:
    X[col] = X[col].replace({'No internet service':'No'})

cat_columns = X.select_dtypes(include='object').columns.to_list()
num_columns = X.select_dtypes(include='number').columns.to_list()

cat_columns1 = X.columns[(X == 'No').any()].to_list() #All these features have a 'No' value which we can target and make 0
cat_columns2 = list(set(X.select_dtypes(include='object').columns)-set(cat_columns1))  # We can drop the first column for these features

drop_no = ['No'] * len(cat_columns1)

transformer = [('OH1', OneHotEncoder(sparse_output=False, drop=drop_no), cat_columns1), # One-Hot Encode and drop 'No' value features 
                ('OH2', OneHotEncoder(sparse_output=False, drop='first'), cat_columns2), # One-Hot Encode and drop first column
                ('scaler', MinMaxScaler(), num_columns), # Scale Numerical columns to lie between 0 and 1
                ]

preprocessor = ColumnTransformer(transformer)

preprocessor.fit(X)
# Export preprocessor
preprocessor_path = Path.cwd() / 'telco_customer_churn' / 'serialization' / 'preprocessor.pickle'
try:
    with open(preprocessor_path, 'wb') as f:
        pickle.dump(preprocessor, f)
        print("Preprocessor saved sucessfully...")
except Exception as e:
    print("Error wile saving preprocessor : ", e)

X_preprocessed = preprocessor.transform(X)

feature_names = []
for name, trans, column in preprocessor.transformers_:
    if hasattr(trans, 'get_feature_names_out'):
        feature_names.extend(trans.get_feature_names_out(column))
    else:
        feature_names.extend(column)

X_preprocessed = pd.DataFrame(X_preprocessed, columns=feature_names)

# Oversampling
smote = SMOTE(k_neighbors=7, sampling_strategy=0.45, random_state=1)
X_resampled, y_resampled = smote.fit_resample(X_preprocessed, y)

# Rename features
columns = X_resampled.columns.to_list()
columns = [col.replace("_Yes", "") for col in columns]
columns = ['gender' if col[:6]=='gender' else col for col in columns]
columns = ['No phone service' if col[-16:]=='No phone service' else col for col in columns]

column_rename = dict(zip(X_resampled.columns, columns))
X_resampled.rename(columns=column_rename, inplace=True)

# Export engineered data
data_engineered = pd.concat([X_resampled, y_resampled], axis=1)
sav_path = Path.cwd() / 'telco_customer_churn' / 'serialization' / 'engineered_data.csv'
data_engineered.to_csv(sav_path, index=False, header=True, encoding='utf-8')

# Modeling
model = LogisticRegression(C= 1, max_iter=30, penalty='l1', solver='liblinear', random_state=1)
model.fit(X_resampled, y_resampled)

# Export trained model
model_path = Path.cwd() / 'telco_customer_churn' / 'serialization' / 'churn_predictor.pickle'
try:
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
        print("Model saved sucessfully...")
except Exception as e:
    print("Error wile saving model : ", e)

# column_names = X_resampled.columns.to_list()
columns_path = Path.cwd() / 'telco_customer_churn' / 'serialization' / 'column_rename.json'
columns_dict = {'feature_names': feature_names,
                'column_rename' : column_rename}
try:
    with open(columns_path, "w") as f:
        f.write(json.dumps(columns_dict))
    print("Coulmns list saved successfully")
except Exception as e:
    print("Error while saving columns :", e)

print(X_resampled.columns.to_list())