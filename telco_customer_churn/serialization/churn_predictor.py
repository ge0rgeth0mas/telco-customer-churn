# Import Basic Libraries
import pandas as pd
import numpy as np
from pathlib import Path

# Import ML libraries
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import AdaBoostClassifier

# Get data
path = Path.cwd() / 'telco_customer_churn' / 'resources' / 'Telco-Customer-Churn.csv' # To ensure path works in both WIndows and Mac
data = pd.read_csv(path)

# Data Preprocessing
data['TotalCharges']=pd.to_numeric(data.TotalCharges, errors='coerce') # 'TotalCharges' feature is of 'Object' type and has empthy spaces instead of missing values
data.drop(data[data.TotalCharges.isna()].index, axis=0, inplace=True) # Remove observations woth missing values
data.drop('customerID', axis=1, inplace=True)

# Create X (Features) and y (Target)
X=data.copy()
X.drop(['PhoneService', 'TotalCharges'], axis=1, inplace=True)

internet_addon_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

for col in internet_addon_services: # Change 'No internet service'to 'No', as 'InterenetService' Feature captures if they have internet
    if col in X.columns:
        X[col] = X[col].replace({'No internet service':'No'})

cat_columns = X.select_dtypes(include='object').columns
num_columns = X.select_dtypes(include='number').columns

y = data['Churn'].replace({'Yes':1, "No":0}) # Convert Yes/No to 1/0 in Target

cat_columns1 = X.columns[(X == 'No').any()].to_list() #All these features have a 'No' value which we can target and make 0
cat_columns2 = cat_columns.drop(cat_columns1)  # We can drop the first column for these features

drop_no = ['No'] * len(cat_columns1)

transformer = [('OH1', OneHotEncoder(sparse_output=False, drop=drop_no), cat_columns1), # One-Hot Encode and drop 'No' value features 
            ('OH2', OneHotEncoder(sparse_output=False, drop='first'), cat_columns2), # One-Hot Encode and drop first column
            ('scaler', MinMaxScaler(), num_columns)] # Scale Numerical columns to lie between 0 and 1
preprocessor = ColumnTransformer(transformer)

X = preprocessor.fit_transform(X)

feature_names = []
for name, trans, column in preprocessor.transformers_:
    if hasattr(trans, 'get_feature_names_out'):
        feature_names.extend(trans.get_feature_names_out(column))
    else:
        feature_names.extend(column)

X = pd.DataFrame(X, columns=feature_names)

smote=SMOTE(sampling_strategy='auto', random_state=1)

X, y = smote.fit_resample(X, y)

classifier = AdaBoostClassifier(n_estimators=250, learning_rate=1.3)

classifier.fit(X,y)

