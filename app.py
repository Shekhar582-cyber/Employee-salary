import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained pipeline (scaler + model)
try:
    model_pipeline = joblib.load("best_model.pkl")
    # Load the income encoder specifically
    income_encoder = joblib.load("label_encoder_income.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
except FileNotFoundError:
    st.error("Model files not found. Please run model_training.py first to generate 'best_model.pkl', 'label_encoder_income.pkl', and 'feature_columns.pkl' along with other label encoders.")
    st.stop()

st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")

st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

# Sidebar inputs (these must match your training feature columns and original data types)
st.sidebar.header("Input Employee Details")

# Define the unique values for categorical features from your original data analysis
# These lists should ideally be derived dynamically from the fitted encoders or a configuration.
# For now, manually ensure they match the categories seen during training.
workclass_options = ['Private', 'Self-emp-not-inc', 'Local-gov', 'Others', 'State-gov', 'Self-emp-inc', 'Federal-gov']
marital_status_options = ['Never-married', 'Married-civ-spouse', 'Divorced', 'Married-spouse-absent', 'Separated', 'Married-AF-spouse', 'Widowed']
occupation_options = ['Machine-op-inspct', 'Farming-fishing', 'Protective-serv', 'Craft-repair', 'Other-service', 'Tech-support', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Transport-moving', 'Priv-house-serv', 'Armed-Forces', 'Others'] # 'Others' for '?'
relationship_options = ['Own-child', 'Husband', 'Not-in-family', 'Unmarried', 'Wife', 'Other-relative']
race_options = ['Black', 'White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other']
gender_options = ['Male', 'Female']
native_country_options = [
    'United-States', 'Peru', 'Guatemala', 'Mexico', 'Dominican-Republic',
    'Ireland', 'Germany', 'Philippines', 'Thailand', 'Haiti', 'El-Salvador',
    'Puerto-Rico', 'France', 'Columbia', 'Hungary', 'India', 'Jamaica',
    'China', 'Cuba', 'Iran', 'Honduras', 'Nicaragua', 'Taiwan', 'Canada',
    'Poland', 'England', 'South', 'Japan', 'Italy', 'Portugal', 'Vietnam',
    'Hong', 'Ecuador', 'Scotland', 'Trinadad&Tobago', 'Outlying-US(GUAM-USVI-etc)',
    'Cambodia', 'Laos', 'Yugoslavia', 'Greece', 'Others' # Added 'Others' for '?' and ensured no duplicates from manual list.
]

age = st.sidebar.slider("Age", 17, 75, 30) # Adjusted range based on outlier removal
workclass = st.sidebar.selectbox("Workclass", workclass_options)
fnlwgt = st.sidebar.number_input("Final Weight (fnlwgt)", min_value=10000, max_value=1500000, value=200000)
educational_num = st.sidebar.slider("Educational Number", 5, 16, 9) # Adjusted range
marital_status = st.sidebar.selectbox("Marital Status", marital_status_options)
occupation = st.sidebar.selectbox("Occupation", occupation_options)
relationship = st.sidebar.selectbox("Relationship", relationship_options)
race = st.sidebar.selectbox("Race", race_options)
gender = st.sidebar.selectbox("Gender", gender_options)
capital_gain = st.sidebar.number_input("Capital Gain", min_value=0, max_value=100000, value=0)
capital_loss = st.sidebar.number_input("Capital Loss", min_value=0, max_value=4500, value=0)
hours_per_week = st.sidebar.slider("Hours per week", 1, 99, 40) # Max hours from data, not 80
native_country = st.sidebar.selectbox("Native Country", native_country_options)

# Create a DataFrame for the input
input_data = pd.DataFrame([[
    age, workclass, fnlwgt, educational_num, marital_status, occupation,
    relationship, race, gender, capital_gain, capital_loss, hours_per_week,
    native_country
]], columns=feature_columns)

# Apply Label Encoding to the input data
categorical_cols_for_encoding = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']
for col in categorical_cols_for_encoding:
    try:
        le = joblib.load(f"label_encoder_{col}.pkl")
        # Transform the single input value
        input_data[col] = le.transform(input_data[col])
    except Exception as e:
        st.error(f"Error loading or transforming encoder for column '{col}': {e}")
        st.stop()


st.write("### 🔎 Input Data")
st.write(input_data)

# Predict button
if st.button("Predict Salary Class"):
    try:
        prediction_numeric = model_pipeline.predict(input_data)
        prediction_label = income_encoder.inverse_transform(prediction_numeric)
        st.success(f"✅ Predicted Income Class: **{prediction_label[0]}**")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.info("Please ensure all input values are valid and the model files are correctly loaded.")

# Batch prediction
st.markdown("---")
st.markdown("#### 📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    st.write("Uploaded data preview:", batch_data.head())

    # Apply the same preprocessing steps to the batch data
    batch_data.replace('?', 'Others', inplace=True)
    batch_data = batch_data[batch_data['workclass'] != 'Without-pay']
    batch_data = batch_data[batch_data['workclass'] != 'Never-worked']
    batch_data = batch_data[(batch_data['age'] <= 75) & (batch_data['age'] >= 17)]
    batch_data = batch_data[(batch_data['educational-num'] <= 16) & (batch_data['educational-num'] >= 5)]
    batch_data = batch_data.drop(columns=['education'])

    for col in categorical_cols_for_encoding:
        try:
            le = joblib.load(f"label_encoder_{col}.pkl")
            # Handle potential new categories in uploaded batch data
            # For unseen categories, LabelEncoder will raise an error.
            # A common approach is to treat them as 'Others' or a specific placeholder.
            # Here, we'll try to transform, and if it fails (unseen category),
            # we'll map it to the 'Others' category if 'Others' is one of the learned classes.
            # Otherwise, it might be safer to drop such rows or use a more advanced imputation.
            
            # Create a mapping dictionary for inverse transform lookup and default for unseen
            mapping = {cls: idx for idx, cls in enumerate(le.classes_)}
            
            # Map values, defaulting to 'Others' if available, otherwise NaN or a placeholder
            if 'Others' in le.classes_:
                default_val = mapping['Others']
            else:
                default_val = -1 # Or choose another appropriate handling for unseen

            batch_data[col] = batch_data[col].apply(lambda x: mapping.get(x, default_val))

        except Exception as e:
            st.error(f"Error loading or transforming encoder for batch column '{col}': {e}")
            st.stop()

    # Ensure columns are in the same order as trained features
    batch_data = batch_data[feature_columns]

    try:
        batch_predictions_numeric = model_pipeline.predict(batch_data)
        batch_predictions_label = income_encoder.inverse_transform(batch_predictions_numeric)
        batch_data['Predicted Income Class'] = batch_predictions_label

        st.write("✅ Predictions (first 5 rows):")
        st.write(batch_data.head())

        csv = batch_data.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions CSV", csv, file_name='predicted_income_classes.csv', mime='text/csv')
    except Exception as e:
        st.error(f"An error occurred during batch prediction: {e}")
        st.info("Please ensure the uploaded CSV has the correct columns and data format.")