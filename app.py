import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Load the saved model pipeline
stacking_pipeline = joblib.load(open('stacking_pipeline_model.pkl', 'rb'))

# Set up the Streamlit UI
st.set_page_config(page_title="Stroke Prediction", layout="wide")
st.title("Stroke Prediction Model")

# Add a subtitle and description
st.subheader("Predict the likelihood of a stroke based on patient data.")
st.write("Fill in the details below to assess the risk of a stroke.")

# Customize input styling
st.markdown("""
<style>
    .stButton>button {
        background-color: #FF6347;
        color: white;
        font-size: 20px;
        height: 50px;
        border-radius: 10px;
    }
    .stButton>button:hover {
        background-color: #FF4500;
    }
    .stSelectbox select, .stNumberInput input {
        font-size: 16px;
        padding: 10px;
    }
    .stTitle, .stSubheader {
        text-align: center;
        color: #2e3a59;
    }
    .stText {
        font-size: 18px;
        font-weight: 500;
        color: #4d4d4d;
    }
</style>
""", unsafe_allow_html=True)

# Define the input fields in a single column
age = st.number_input("Age", min_value=0, max_value=120, value=25)
gender = st.selectbox("Gender", options=['Male', 'Female', 'Other'])

# Define the input fields with 'Yes' and 'No' options
hypertension = st.selectbox("Hypertension", options=["Yes", "No"], help="Select 'Yes' for Hypertension, 'No' for No Hypertension")
heart_disease = st.selectbox("Heart Disease", options=["Yes", "No"], help="Select 'Yes' for Heart Disease, 'No' for No Heart Disease")

# Convert the 'Yes'/'No' inputs to numerical format for model prediction
hypertension = 1 if hypertension == "Yes" else 0
heart_disease = 1 if heart_disease == "Yes" else 0

avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, max_value=300.0, value=100.0)

work_type = st.selectbox("Work Type", options=['Private', 'Self-employed', 'Govt_job', 'Never_worked'])
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=22.0)
residence_type = st.selectbox("Residence Type", options=['Rural', 'Urban'])

# Convert categorical inputs to numerical format
gender_dict = {'Male': 0, 'Female': 1, 'Other': -1}
work_type_dict = {'Private': 0, 'Self-employed': 1, 'Govt_job': 2, 'children': -1, 'Never_worked': -2}
residence_type_dict = {'Rural': 0, 'Urban': 1}

gender = gender_dict[gender]
work_type = work_type_dict[work_type]
residence_type = residence_type_dict[residence_type]

df = pd.DataFrame([[gender, age, hypertension, heart_disease, work_type, residence_type, avg_glucose_level, bmi]],
                  columns=['gender', 'age', 'hypertension', 'heart_disease', 'work_type', 'residence_type','avg_glucose_level', 'bmi'])

# Function for outlier treatment
def outlier_treating(data, var):
    df = data.copy()
    def outlier_detector(data):
        outliers = []
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        IQR = q3 - q1
        lb = q1 - (IQR * 1.5)
        ub = q3 + (IQR * 1.5)
        for i, j in enumerate(data):
            if j < lb or j > ub:
                outliers.append(i)
        return outliers
    for i in var:
        out_var = outlier_detector(df[i])
        df.loc[out_var, i] = np.median(df[i])
    return df

# Assuming 'df' is your dataset, load it from the source before preprocessing  # Replace with your actual data
var = list(df.select_dtypes(include=['float64']).columns)
df = outlier_treating(df, var)


# Data encoding
df['gender'] = df['gender'].replace({'Male': 0, 'Female': 1, 'Other': -1}).astype(np.uint8)
df['residence_type'] = df['residence_type'].replace({'Rural': 0, 'Urban': 1}).astype(np.uint8)
df['work_type'] = df['work_type'].replace({'Private': 0, 'Self-employed': 1, 'Govt_job': 2, 'children': -1, 'Never_worked': -2}).astype(np.uint8)

# Arrange the input data for prediction
input_data = pd.DataFrame([[gender, age, hypertension, heart_disease, work_type, avg_glucose_level, bmi]],
                          columns=['gender', 'age', 'hypertension', 'heart_disease', 'work_type', 'avg_glucose_level', 'bmi'])

# Prediction button with custom style
if st.button("Predict Stroke Risk"):
    # Make prediction
    prediction = stacking_pipeline.predict(input_data)
    risk = "High" if prediction[0] == 1 else "Low"
    
    # Display the result with some flair
    st.markdown(f"<h3 style='text-align: center; color: {'#FF6347' if risk == 'High' else '#32CD32'};'>Predicted Stroke Risk: {risk}</h3>", unsafe_allow_html=True)
    st.write(f"Based on the provided data, the likelihood of having a stroke is **{risk}**.")

    # Add some information about stroke risk for more context
    if risk == "High":
        st.write("Please consult a healthcare professional for further evaluation.")
    else:
        st.write("Keep maintaining a healthy lifestyle to reduce the risk.")

# Footer for app customization
st.markdown("""
<hr>
<p style="text-align: center;">Powered by Streamlit</p>
""", unsafe_allow_html=True)
