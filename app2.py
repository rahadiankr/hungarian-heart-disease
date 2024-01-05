import streamlit as st
import pickle as pkl
import pandas as pd

# Page Setup
st.set_page_config(page_title="Hungarian Heart Disease", page_icon=":heart:")

# Page Title
st.title("🫀 Hungarian Heart Disease")
st.write("Predict heart disease using machine learning models.")
st.write("This app is using Hungarian Heart Disease Dataset from UCI Machine Learning Repository. The dataset contains 294 observations and 14 attributes. The target variable is the presence or absence of heart disease in the patient. The dataset can be found [here](https://archive.ics.uci.edu/ml/datasets/heart+disease).")

# Model Setup
def load_model(model_file):
    loaded_model = pkl.load(open(model_file, "rb"))
    return loaded_model

# Selet Model
model_select = st.selectbox("Select Model", options=["KNN", "Random Forest", "XGBoost"], index=None, placeholder="Chose an Option", format_func= lambda x: "KNN" if x == "KNN" else "Random Forest" if x == "Random Forest" else "XGBoost")

# Load Model
if model_select == "KNN":
    model = load_model("models/model_knn.pkl")
elif model_select == "Random Forest":
    model = load_model("models/model_rf.pkl")
elif model_select == "XGBoost":
    model = load_model("models/model_xgb.pkl")
else:
    st.warning('⚠️ Please select a model.')
    st.stop()

# Model Accuracy
if model_select == "KNN":
    model_acc = 93.0
elif model_select == "Random Forest":
    model_acc = 92.0
elif model_select == "XGBoost":
    model_acc = 90.9
    
# Model Info
st.caption("Model: " + model_select)
st.caption("Model Akurasi: " + str(model_acc))

# Tabs Setup for Single Prediction and Multi Prediction
tab1, tab2 = st.tabs(["Single Prediction", "Multi Prediction"])

# Single Prediction
with tab1:
    # Tab 1 Header
    st.header("Single Prediction")
    st.write("Predict heart disease for a single patient")
    
    # Inputs Features
    st.subheader("User Input Features")
    # Columns View
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=0, max_value=100, value=None, help="Represents the age of the patient in years.")
        sex = st.selectbox("Sex", options=[0, 1], index=None, placeholder="Chose an Option", format_func= lambda x: "Male" if x == 1 else "Female", help="Indicates the gender of the patient. Male is represented by 1, and female by 0.")
        cp = st.selectbox("Chest Pain Type", options=[1,2,3,4], index=None, placeholder="Chose an Option", format_func= lambda x: "Typical Angina" if x == 1 else "Atypical Angina" if x == 2 else "Non-Anginal Pain" if x == 3 else "Asymptomatic", help="Describes the type of chest pain experienced by the patient. Different types of chest pain can be indicative of various heart conditions.")
        trestbps = st.number_input("Resting Blood Pressure", min_value=0, max_value=200, value=None, help="Represents the resting blood pressure of the patient in mmHg on admission to the hospital.")
        chol = st.number_input("Serum Cholestoral in mg/dl", min_value=0, max_value=600, value=None, help="Represents the serum cholesterol level of the patient in mg/dl.")
    with col2:
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], index=None, placeholder="Chose an Option", format_func= lambda x: "True" if x == 1 else "False", help="Indicates whether the patient's fasting blood sugar was greater than 120 mg/dl. A value of 1 indicates that it was greater than 120 mg/dl, and a value of 0 indicates that it was less than 120 mg/dl.")
        restecg = st.selectbox("Resting Electrocardiographic Results", options=[0,1,2], index=None, placeholder="Chose an Option", format_func= lambda x: "Normal" if x == 0 else "ST-T Wave Abnormality" if x == 1 else "Left Ventricular Hypertrophy", help="Represents the resting electrocardiographic results of the patient. A value of 0 indicates normal, a value of 1 indicates ST-T wave abnormality, and a value of 2 indicates left ventricular hypertrophy.")
        thalach = st.number_input("Maximum Heart Rate Achieved", min_value=0, max_value=300, value=None, help="Represents the maximum heart rate achieved by the patient.")
        exang = st.selectbox("Exercise Induced Angina", options=[0,1], index=None, placeholder="Chose an Option", format_func= lambda x: "Yes" if x == 1 else "No", help="Indicates whether the patient experienced exercise induced angina. A value of 1 indicates that they did, and a value of 0 indicates that they did not.")
        oldpeak = st.number_input("Oldpeak", min_value=0, max_value=10, value=None, help="Represents the ST depression induced by exercise relative to rest.")
        
    # Inputs to dataframe 
    data = [{
        "Age": age,
        "Sex": sex,
        "Cp": cp,
        "Trestbps": trestbps,
        "Chol": chol,
        "Fbs": fbs,
        "Restecg": restecg,
        "Thalach": thalach,
        "Exang": exang,
        "Oldpeak": oldpeak,
        }]
    data = pd.DataFrame(data)
    
    # Data check if Null stop
    if data.isnull().sum().sum() > 0:
        st.warning('⚠️ Please fill in all the data.')
        st.stop()
    
        
    # Predict Button
    if st.button("Predict", type="primary", key="single-prediction", disabled=False):
        st.subheader("Data Input:")
        # Show data entry from user input
        st.write(data)
        # Predict
        st.subheader("Prediction:")
        prediction = model.predict(data)[0]
        if prediction == 0:
            st.success(':green[No Heart Disease]', icon="✅")
        else:
            st.error(':red[Heart Disease]', icon="🔴")
    # # Reset Button
    # if st.button("Reset", type="secondary", key="single-prediction-reset"):
    #     st.write("")
        
with tab2:
    st.subheader("Multi Prediction")
    st.write("Predict heart disease for multiple patients")
    
    # Upload File
    uploaded_file = st.file_uploader("Upload File", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        # Predict Button
        if st.button("Predict", type="primary", key="multi-prediction"):
            st.subheader("Data Input:")
            # Show data entry from user input
            st.write(data)
            # Predict
            prediction = model.predict(data)
            prediction = pd.DataFrame(prediction)
            prediction = prediction.replace({0: "No Heart Disease", 1: "Heart Disease", 2: "Heart Disease", 3: "Heart Disease", 4: "Heart Disease"})
            st.subheader("Prediction:")
            st.write(prediction)
        # Reset Button
        if st.button("Reset", type="secondary", key="multi-prediction-reset"):
            st.write("")

# Footer
st.markdown("---")
st.write("Created by 👨‍💻 [rahadiankr](https://github.com/rahadiankr/hungarian-heart-disease)")