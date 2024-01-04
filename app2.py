import streamlit as st
import pickle as pkl
import pandas as pd

# Page Setup
st.set_page_config(page_title="Hungarian Heart Disease", page_icon=":heart:")

# Page Title
st.title("Hungarian Heart Disease")
st.write("Predict heart disease using machine learning")

# Model Setup
def load_model(model_file):
    loaded_model = pkl.load(open(model_file, "rb"))
    return loaded_model

# Selet Model
model_select = st.selectbox("Select Model", options=["KNN", "Random Forest", "XGBoost"], index=0)

# Load Model
if model_select == "KNN":
    model = load_model("models/model_knn.pkl")
elif model_select == "Random Forest":
    model = load_model("models/model_rf.pkl")
elif model_select == "XGBoost":
    model = load_model("models/model_xgb.pkl")
    

tab1, tab2 = st.tabs(["Single Prediction", "Multi Prediction"])
with tab1:
    st.header("Single Prediction")
    st.write("Predict heart disease for a single patient")
    
    # Inputs Features
    st.subheader("User Input Features")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=0, max_value=100, value=0, )
        sex = st.selectbox("Sex", options=[0, 1], index=None, placeholder="Chose an Option", format_func= lambda x: "Male" if x == 1 else "Female")
        cp = st.selectbox("Chest Pain Type", options=[1,2,3,4], index=None, placeholder="Chose an Option", format_func= lambda x: "Typical Angina" if x == 1 else "Atypical Angina" if x == 2 else "Non-Anginal Pain" if x == 3 else "Asymptomatic")
        trestbps = st.number_input("Resting Blood Pressure", min_value=0, max_value=200, value=0)
        chol = st.number_input("Serum Cholestoral in mg/dl", min_value=0, max_value=600, value=0)
    with col2:
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], index=None, placeholder="Chose an Option", format_func= lambda x: "True" if x == 1 else "False")
        restecg = st.selectbox("Resting Electrocardiographic Results", options=[0,1,2], index=None, placeholder="Chose an Option", format_func= lambda x: "Normal" if x == 0 else "ST-T Wave Abnormality" if x == 1 else "Left Ventricular Hypertrophy")
        thalach = st.number_input("Maximum Heart Rate Achieved", min_value=0, max_value=300, value=0)
        exang = st.selectbox("Exercise Induced Angina", options=[0,1], index=None, placeholder="Chose an Option", format_func= lambda x: "Yes" if x == 1 else "No")
        oldpeak = st.number_input("Oldpeak", min_value=0, max_value=10, value=0)

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
        
    # Predict Button
    if st.button("Predict", type="primary", key="single-prediction"):
        st.subheader("Data Input:")
        # Show data entry from user input
        st.write(data)
        # Predict
        prediction = model.predict(data)[0]
        if prediction == 0:
            prediction = ":ok: :green[No Heart Disease]"
        else:
            prediction = ":warning: :red[Heart Disease]"
        st.subheader("Prediction:")
        st.subheader(prediction)
        
    if st.button("Reset", type="secondary", key="single-prediction-reset"):
        st.write("")
        
    
        
    
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
            
        if st.button("Reset", type="secondary", key="multi-prediction-reset"):
            st.write("")
            
    

    







