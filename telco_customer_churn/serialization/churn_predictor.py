import pandas as pd
import numpy as np
from pathlib import Path


path = Path.cwd() / 'telco_customer_churn' / 'resources' / 'Telco-Customer-Churn.csv'
data = pd.read_csv(path)
print(data.head())