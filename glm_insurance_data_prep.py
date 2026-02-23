# GLM INSURANCE ANALYTICS DATA PREP FOR MODELING


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Load data
df = pd.read_csv('insurance_claims.csv')

# Basic inspection
print("Dataset Shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nBasic statistics:")
print(df.describe())
print("\nMissing values:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Data Prep Pipeline

# Handle missing values
df['vehicle_value'] = df.groupby('vehicle_type')['vehicle_value'].transform(
    lambda x: x.fillna(x.median())
)
df['policy_tenure'].fillna(df['policy_tenure'].median(), inplace=True)

# Create CA-insurance features before encoding
df['age_band'] = pd.cut(df['age'], 
                        bins=[0, 25, 35, 50, 65, 100],
                        labels=['16-25', '26-35', '36-50', '51-65', '65+'])

df['vehicle_age_category'] = pd.cut(df['vehicle_age'],
                                    bins=[-1, 3, 7, 15, 100],
                                    labels=['New', 'Mid-Age', 'Old', 'Very Old'])

df['premium_segment'] = 'Standard'
df.loc[(df['coverage_type'] == 'Premium') & (df['vehicle_value'] > 40000), 'premium_segment'] = 'High'
df.loc[(df['coverage_type'] == 'Basic') & (df['vehicle_value'] < 20000), 'premium_segment'] = 'Low'

df['high_deductible'] = (df['deductible'] >= 1000).astype(int)

# One-hot encoding all categorical variables 
cat_vars = ['gender', 'marital_status', 'vehicle_type', 'coverage_type', 
            'region', 'state', 'claim_type', 'at_fault', 'weather_condition', 
            'time_of_day', 'age_band', 'vehicle_age_category', 'premium_segment']

df_encoded = pd.get_dummies(df, columns=cat_vars, drop_first=True, dtype=int)

# 4. Create features and target
X = df_encoded.drop(['policy_id', 'claim_amount'], axis=1)
y = df_encoded['claim_amount']

# 5. Verify all elements are numeric
assert X.select_dtypes(include=['int64', 'float64']).shape[1] == X.shape[1], "Not all features are numeric!"
