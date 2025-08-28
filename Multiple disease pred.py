# -*- coding: utf-8 -*-
"""
Created on Wed May 14 12:22:46 2025

@author: Jeela
"""
import pickle #to load saved models 
import streamlit as st
from streamlit_option_menu import option_menu #for creating side bars

#Loading the saved models
diabetes_model=pickle.load(open("C:/Users/Jeela/OneDrive/Desktop/Multiple Diseases Prediction/cp2_models/diabetes_model.sav","rb"))
scaler_diabetes=pickle.load(open("C:/Users/Jeela/OneDrive/Desktop/Multiple Diseases Prediction/cp2_models/scaler_diabetes.sav","rb"))

heart_disease_model = pickle.load(open("C:/Users/Jeela/OneDrive/Desktop/Multiple Diseases Prediction/cp2_models/heart_disease_model.sav", "rb"))
scaler_heart = pickle.load(open("C:/Users/Jeela/OneDrive/Desktop/Multiple Diseases Prediction/cp2_models/scaler_heart.sav", "rb"))



parkinsons_model=pickle.load(open("C:/Users/Jeela/OneDrive/Desktop/Multiple Diseases Prediction/cp2_models/parkinson's_disease_model.sav","rb"))


#sidebar for navigate

with st.sidebar:
    selected = option_menu("Multiple Diseases Prediction System",
                         ["Diabetes Prediction",
                          "Heart Disease Prediction",
                          "Parkinsons Prediction"],
                         
                         icons = ["activity","heart","person"],
                         
                         default_index=0)
 
#Diabetes Prediction Page
if selected == "Diabetes Prediction":

    # Page title
    st.title("Diabetes Disease Prediction using ML")

    # Display image with full container width
    st.image("images/diabetic.jpeg", use_container_width=True)

    # Getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("<h5 style='font-size:18px;'>Number of Pregnancies</h5>", unsafe_allow_html=True)
        Pregnancies = st.text_input("", key="pregnancies")

        st.markdown("<h5 style='font-size:18px;'>Skin Thickness Value</h5>", unsafe_allow_html=True)
        SkinThickness = st.text_input("", key="skin_thickness")

        st.markdown("<h5 style='font-size:18px;'>Diabetes Pedigree Function Value</h5>", unsafe_allow_html=True)
        DiabetesPedigreeFunction = st.text_input("", key="diabetes_pedigree_function")

    with col2:
        st.markdown("<h5 style='font-size:18px;'>Glucose Level</h5>", unsafe_allow_html=True)
        Glucose = st.text_input("", key="glucose")

        st.markdown("<h5 style='font-size:18px;'>Insulin Level</h5>", unsafe_allow_html=True)
        Insulin = st.text_input("", key="insulin")

        st.markdown("<h5 style='font-size:18px;'>Age</h5>", unsafe_allow_html=True)
        Age = st.text_input("", key="age")

    with col3:
        st.markdown("<h5 style='font-size:18px;'>Blood Pressure Value</h5>", unsafe_allow_html=True)
        BloodPressure = st.text_input("", key="blood_pressure")

        st.markdown("<h5 style='font-size:18px;'>BMI Value</h5>", unsafe_allow_html=True)
        BMI = st.text_input("", key="bmi")

    # Code for prediction
    diab_diagnosis = ""

    # Creating a button for prediction
    if st.button("Diabetes test Result"):
        try:
            # Convert inputs to float
            input_data = [
                float(Pregnancies), float(Glucose), float(BloodPressure),
                float(SkinThickness), float(Insulin), float(BMI),
                float(DiabetesPedigreeFunction), float(Age)
            ]

            # Apply scaler
            input_data_scaled = scaler_diabetes.transform([input_data])

            # Make prediction
            diab_prediction = diabetes_model.predict(input_data_scaled)

            # Show result
            if diab_prediction[0] == 1:
                diab_diagnosis = "The Person is Diabetic"
            else:
                diab_diagnosis = "The Person is Not Diabetic"

            st.success(diab_diagnosis)

        except ValueError:
            st.error("Please enter valid numeric values in all input fields.")


              
# Heart Disease Prediction Page    
if selected == "Heart Disease Prediction":
    
    st.title("Heart Disease Prediction using ML")
    st.image("images/Heart.jpeg", use_container_width=True)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        age = st.text_input("Age")
        trestbps = st.text_input("Resting Blood Pressure")
        restecg = st.text_input("ECG Results")

    with col2:
        sex = st.text_input("Gender (0 = female, 1 = male)")
        chol = st.text_input("Cholesterol Level")
        thalach = st.text_input("Maximum Heart Rate")

    with col3:
        cp = st.text_input("Chest Pain Type")
        fbs = st.text_input("Fasting Blood Sugar (0 or 1)")
        exang = st.text_input("Exercise-induced Angina (0 or 1)")

    with col4:
        oldpeak = st.text_input("ST Depression after Exercise")
        slope = st.text_input("Slope of ST Segment")
        ca = st.text_input("Number of Major Vessels")

    with col5:
        thal = st.text_input("Thalassemia Type")

    heart_diagnosis = ""

    if st.button("Heart Disease Test Result"):
        try:
            # Convert all inputs to float (or int where needed)
            input_data = [
                float(age), float(sex), float(cp), float(trestbps), float(chol),
                float(fbs), float(restecg), float(thalach), float(exang),
                float(oldpeak), float(slope), float(ca), float(thal)
            ]

            input_data_scaled = scaler_heart.transform([input_data])
            prediction = heart_disease_model.predict(input_data_scaled)

            if prediction[0] == 1:
                heart_diagnosis = "The person has heart disease."
            else:
                heart_diagnosis = "The person does not have heart disease."

            st.success(heart_diagnosis)

        except ValueError:
            st.error("Please enter valid numeric values in all fields.")
        except Exception as e:
            st.error(f"Error: {e}")

# Parkinson's Disease Prediction Page
if selected == "Parkinsons Prediction":

    st.title("Parkinson's Disease Prediction using ML")
    st.image("images/parkinsons.jpeg", use_container_width=True)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.text_input("MDVP:Fo(Hz)")
        rap = st.text_input("MDVP:RAP")
        shimmer_apq3 = st.text_input("Shimmer:APQ3")
        hnr = st.text_input("HNR")
        d2 = st.text_input("D2")

    with col2:
        fhi = st.text_input("MDVP:Fhi(Hz)")
        ppq = st.text_input("MDVP:PPQ")
        shimmer_apq5 = st.text_input("Shimmer:APQ5")
        rpde = st.text_input("RPDE")
        ppe = st.text_input("PPE")

    with col3:
        flo = st.text_input("MDVP:Flo(Hz)")
        ddp = st.text_input("Jitter:DDP")
        mdvp_apq = st.text_input("MDVP:APQ")
        dfa = st.text_input("DFA")

    with col4:
        jitter_percent = st.text_input("MDVP:Jitter(%)")
        shimmer = st.text_input("MDVP:Shimmer")
        dda = st.text_input("Shimmer:DDA")
        spread1 = st.text_input("spread1")

    with col5:
        jitter_abs = st.text_input("MDVP:Jitter(Abs)")
        shimmer_db = st.text_input("MDVP:Shimmer(dB)")
        nhr = st.text_input("NHR")
        spread2 = st.text_input("spread2")

    parkinsons_diagnosis = ""

    if st.button("Parkinson's Test Result"):
        try:
            # Collect and clean all inputs
            inputs = [
                fo, fhi, flo, jitter_percent, jitter_abs, rap,
                ppq, ddp, shimmer, shimmer_db, shimmer_apq3,
                shimmer_apq5, mdvp_apq, dda, nhr, hnr,
                rpde, dfa, spread1, spread2, d2, ppe
            ]

            # Check for blanks
            if any(i.strip() == '' for i in inputs):
                st.error("Please fill in **all fields**. One or more fields are empty.")
            else:
                # Convert to float
                input_data = [float(i.strip()) for i in inputs]

                prediction = parkinsons_model.predict([input_data])

                if prediction[0] == 1:
                    parkinsons_diagnosis = "The person has Parkinson's Disease."
                else:
                    parkinsons_diagnosis = "The person does not have Parkinson's Disease."

                st.success(parkinsons_diagnosis)

        except ValueError:
            st.error("One or more fields contain **invalid numbers**. Please use only numeric values (e.g., 0.01234).")







    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    