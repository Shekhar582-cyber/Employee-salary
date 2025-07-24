import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
import joblib
import numpy as np

# Load the dataset
try:
    data = pd.read_csv(r"C:\Users\shekh\Downloads\adult 3 (1).csv")
except FileNotFoundError:
    print("Error: 'adult 3 (1).csv' not found. Please make sure the CSV file is in the same directory.")
    exit()

# Data Cleaning and Preprocessing (from your notebook)
data.replace('?', 'Others', inplace=True)

# Remove 'Without-pay' and 'Never-worked' from 'workclass'
data = data[data['workclass'] != 'Without-pay']
data = data[data['workclass'] != 'Never-worked']

# Outlier removal based on your boxplots and code
# For 'age': removed outliers outside 17-75 range
data = data[(data['age'] <= 75) & (data['age'] >= 17)]
# For 'capital-gain': Your boxplot showed many outliers, but you didn't explicitly filter them.
# Given the distribution, it's common to not remove them as they are genuine values for capital gains.
# For 'educational-num': removed outliers outside 5-16 range
data = data[(data['educational-num'] <= 16) & (data['educational-num'] >= 5)]

# Drop redundant 'education' column
data = data.drop(columns=['education'])

# Apply Label Encoding
categorical_cols = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country', 'income']
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le
    # Save each individual LabelEncoder
    joblib.dump(le, f"label_encoder_{col}.pkl") # <--- This line is crucial for saving each encoder

# Separate features (x) and target (y)
x = data.drop(columns=['income'])
y = data['income']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Initialize StandardScaler
# It's good practice to save the scaler too if you plan to scale inputs in Streamlit
scaler = StandardScaler()

# Create a pipeline for the best model (GradientBoostingClassifier)
# It's crucial to scale the data BEFORE training, and the pipeline ensures this consistency.
pipeline = Pipeline([
    ('scaler', scaler),
    ('model', GradientBoostingClassifier())
])

# Train the pipeline
pipeline.fit(X_train, y_train)

# Save the trained pipeline (which includes the scaler and the model)
joblib.dump(pipeline, "best_model.pkl")
joblib.dump(x.columns.tolist(), "feature_columns.pkl") # Save feature column names

print("Model training and saving complete!")
print("Saved best_model.pkl (GradientBoostingClassifier with StandardScaler)")
print("Saved individual label encoders for categorical features.")
print("Saved feature_columns.pkl")