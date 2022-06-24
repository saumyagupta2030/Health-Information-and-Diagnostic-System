from turtle import onclick, width
from xml.etree.ElementTree import SubElement
import pandas as pd
import numpy as np
import pickle
import streamlit as st
import requests
from streamlit_option_menu import option_menu
from PIL import Image
import sys 
import os
#import SessionState


# Remove whitespace from the top of the page and sidebar
    
if 'button_clicked' not in st.session_state:
    st.session_state.button_clicked = False



def callback():
    st.session_state.button_clicked = True


def symptoms():
    st.title("Disease Prediction based on Symptoms")
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        #st.image("/Users/saumyagupta/Desktop/DiseasePredictionSystem/Stroke/Images/Symptoms/BGRemoved/Cough-removebg-preview.png")
        st.image("/Users/saumyagupta/Desktop/DiseasePredictionSystem/Stroke/Images/Symptoms/W:O BG/Cough1-removebg-preview.png", width = 100,caption = "Cough")
    with col2:
        #st.image("/Users/saumyagupta/Desktop/DiseasePredictionSystem/Stroke/Images/Symptoms/BGRemoved/Fever-removebg-preview.png")
        st.image("/Users/saumyagupta/Desktop/DiseasePredictionSystem/Stroke/Images/Symptoms/W:O BG/Fever1-removebg-preview.png", width = 100,caption = "Fever")    
    with col3:
        st.image("/Users/saumyagupta/Desktop/DiseasePredictionSystem/Stroke/Images/Symptoms/W:O BG/MuscleAche1-removebg-preview.png", width = 100, caption= "Muscle or Body Ache")
        #st.image("/Users/saumyagupta/Desktop/DiseasePredictionSystem/Stroke/Images/Symptoms/BGRemoved/MuscleAche-removebg-preview.png")
    with col4:
        st.image("/Users/saumyagupta/Desktop/DiseasePredictionSystem/Stroke/Images/Symptoms/W:O BG/RunnyNose1-removebg-preview.png", width = 100,caption = "Congestion or Runny Nose")
        #st.image("/Users/saumyagupta/Desktop/DiseasePredictionSystem/Stroke/Images/Symptoms/BGRemoved/RunnyNose-removebg-preview.png")
    with col5:
        st.image("/Users/saumyagupta/Desktop/DiseasePredictionSystem/Stroke/Images/Symptoms/W:O BG/Sorethroat1-removebg-preview.png", width = 100,caption = "Sore Throat")
        #st.image("/Users/saumyagupta/Desktop/DiseasePredictionSystem/Stroke/Images/Symptoms/BGRemoved/SoreThroat-removebg-preview.png")
    with col6:
        st.image("/Users/saumyagupta/Desktop/DiseasePredictionSystem/Stroke/Images/Symptoms/W:O BG/ShortnessOfBreadth-removebg-preview.png", width = 100,caption = "Shortness of Breadth")


    symptoms_list = pickle.load(open("/Users/saumyagupta/Desktop/DiseasePredictionSystem/Stroke/symptoms_list.pkl", "rb"))
    options = st.multiselect("Select your symptoms", symptoms_list)
    
    symptoms_df = pd.DataFrame(columns=symptoms_list)
    #print(symptoms_df)
    row = []
    for i in range(0, len(symptoms_list)):
        row.append(0)
    #print(len(row))
    #print(len(symptoms_list))
    #print(options)
    for i in range(0, len(symptoms_list)):
        for option in options:
            if  symptoms_list[i] == option:
                row[i] = 1
            else:
                row[i] = 0
    #print(row)
    model_symptoms = pickle.load(open("/Users/saumyagupta/Desktop/DiseasePredictionSystem/Stroke/Symptoms_model.pkl", "rb"))
    encoder = pickle.load(open("/Users/saumyagupta/Desktop/DiseasePredictionSystem/Stroke/labelEncoder_for_Symptoms.pkl", "rb"))
    
    if st.button('Predict'):   
        if(options == []):
            st.error("No symptoms selected") 
        
        else:
            row = np.array(row).reshape(1, -1)
        
        #test_df = symptoms_df.append(row, ignore_index=True)
            prediction = model_symptoms.predict(row)
            result = encoder.inverse_transform(prediction)
        #res = str("You may have ") +str(result)
        

            st.success("You may have " + result)
    
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")


def stroke_home():
    st.title("Brain Stroke Prediction System")
    image = Image.open("/Users/saumyagupta/Desktop/DiseasePredictionSystem/Stroke/Images/Stroke.jpeg")
    
    st.image(image, caption = None, width = 700)

    st.write("Stroke is a edical emergency that occurs ...")
    
    if (st.button('Make Prediction', key = 'stroke', on_click = callback) or st.session_state.button_clicked):
        stroke()
        

def diabetes_home():
    #callback1()
    st.title("Diabetes Prediction System")
    image = Image.open("/Users/saumyagupta/Desktop/DiseasePredictionSystem/Stroke/Images/Diabetes.jpeg")
    st.image(image, caption = None, width = 600)
    st.write("Diabetes is a edical emergency that occurs ...")

    if((st.button('Make Predictions',  key = 'diabetes', on_click = callback) or st.session_state.button_clicked)):
        diabetes()

def stroke():
    model = pickle.load(open("Stroke_model.pkl", "rb"))
    labelEncoder = pickle.load(open("LabelEncoder.pkl","rb"))

    gender = st.selectbox("Enter the gender", ('Male', 'Female'))
    Age = st.number_input("Enter age ", min_value = 0, max_value = 100)
    Hypertension = st.selectbox("Do you have hypertension?", ('Yes', 'No'))
    Heart_disease = st.selectbox("Do you have heart disease?", ('Yes', 'No'))
    ever_married = st.selectbox("Are you married?", ('Yes', 'No'))
    smoking_status = st.selectbox("Select smoking status", ('formerly smoked', 'never smoked', 'smokes', 'Unknown'))
    work_type = st.selectbox("Select work type", ('Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'))
    Residence_type = st.selectbox("Select residence type", ('Urban' ,'Rural'))
    col1, col2 = st.columns(2)
    with col1:
        avg_glucose_level = st.number_input("Enter average glucose level")
    with col2:
        bmi = st.number_input("Enter BMI")
    if Hypertension == "Yes": 
        Hypertension = 1
    else: 
        Hypertension = 0

    if Heart_disease == "Yes": 
        Heart_disease = 1
    else: 
        Heart_disease = 0

    smoking_status = labelEncoder.fit_transform(np.array(smoking_status).reshape(-1,1))
    gender = labelEncoder.fit_transform(np.array(gender).reshape(-1,1))
    ever_married = labelEncoder.fit_transform(np.array(ever_married).reshape(-1,1))
    Residence_type = labelEncoder.fit_transform(np.array(Residence_type).reshape(-1,1))
    work_type = labelEncoder.fit_transform(np.array(work_type).reshape(-1,1))

    temp = np.concatenate((smoking_status, gender, ever_married, Residence_type, work_type), axis = 0)
    ##print(temp)
    if st.button('Predict'):
        test = [temp[1], Age, Hypertension, Heart_disease, temp[2],temp[4], temp[3], avg_glucose_level, bmi, temp[0]]
        test = np.array(test).reshape(1,-1)

        prediction = model.predict(test)
        #prediction
        if prediction == 1:
            st.error("You may have stroke")
            st.title("You may have stroke")
        else: 
            st.success("No stroke")
            print("No stroke")

def diabetes():
    #st.title("Diabetes Prediction System")
    model_diabetes = pickle.load(open("diabetes_model_RandomForest.pkl", "rb"))

    Age = st.number_input("Enter age ", min_value = 0, max_value = 100)
    bMI_d = st.number_input("Enter BMI", min_value = 0.0, max_value = 60.0, format="%.2f")
    glucose_level = st.number_input("Enter average glucose level", format="%.2f")
    bP = st.number_input("Enter Blood Pressure", min_value = 30, max_value = 250)
    insulin = st.number_input("Enter insulin levels", min_value = 0, max_value = 300)
    pregnancies = st.number_input("Enter number of pregnancies", min_value = 0, max_value = 15)
    SkinThickness = st.number_input("Enter skin thickness", format="%.2f")
    DiabetesPedigreeFunction = st.number_input("Enter likelihood of diabetes based on family history", format="%.3f")

    sample = [pregnancies, glucose_level, bP, SkinThickness, insulin, bMI_d, DiabetesPedigreeFunction, Age]
    sample = np.array(sample).reshape(1, -1)
   
    if st.button('Predict'):

        pred = model_diabetes.predict(sample)
       
        if pred == 1:
            st.error("You may have diabetes")
        else: 
            st.success("No diabates")
            print("No Diabetes")



def heartAttack():
    st.title("Heart Attack Prediction System")
    model = pickle.load(open("Heart_attack_model_KNN.pkl", "rb"))
    Age = st.number_input("Enter age ", min_value = 0, max_value = 120)
    gender = st.selectbox("Enter the gender", ('Male', 'Female'))
    
    Chest_Pain_Type = st.selectbox("Enter chest pain type:",(0, 1, 2, 3))
    st.text("Value 0: typical angina\nValue 1: atypical angina\nValue 2: non-anginal pain\nValue 3: asymptomatic")
    trtbps = st.number_input("Resting blood pressure (in mm Hg)")
    chol  = st.number_input("Cholestoral in mg/dl fetched via BMI sensor")
    fbs = st.selectbox("Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)", (0,1))
    rest_ecg = st.selectbox("Resting electrocardiographic results", ('Normal', 'Having ST-T wave abnormality', 'Showing probable or definite left ventricular hypertrophy by Estes criteria'))
    thalach = st.number_input("Maximum Heart Rate")
    exng = st.selectbox("Exercise induced angina",('Yes', 'No'))
    caa = st.selectbox("Number of major vessels", (0,1,2,3))
    oldpeak = st.number_input("Previous Peak")
    slp = st.selectbox("Slope", (0,1,2))
    thall = st.selectbox("Thall rate", (0,1,2,3))

    if exng == 'Yes':
        exng = 1
    else:
        exng = 0;

    if(rest_ecg == 'Normal'):
        rest_ecg = 0
    elif(rest_ecg == 'Having ST-T wave abnormality'):
        rest_ecg = 1
    else: 
        rest_ecg = 2
    
    if gender == 'Male':
        gender = 1
    else : 
        gender = 0

    sample =[Age, gender, Chest_Pain_Type, trtbps, chol, fbs, rest_ecg, thalach, exng, oldpeak, slp, caa, thall]
    sample = np.array(sample).reshape(1,-1)
    if st.button('Predict'):
        pred = model.predict(sample)

        if pred == 1:
            st.error("You have high risk of heart attack")
        else:
            st.success("No risk of heart attack")


def welcome():
    
    st.title("Health Information & Diagnostic System")
    st.image("/Users/saumyagupta/Desktop/DiseasePredictionSystem/Stroke/Images/dbwh8kv-375c5c96-00bc-4bd7-b57a-b9908074ed18.jpeg", width = 700)
    st.write("")
    st. markdown("<p style='text-align: justify;'>Disease diagnosis is the identification of an health issue, disease, disorder, or other condition that a person may have. Disease diagnoses could be sometimes very easy tasks, while others may be a bit trickier. There are large data sets available; however, there is a limitation of tools that can accurately determine the patterns and make predictions. The traditional methods which are used to diagnose a disease are manual and error-prone. Usage of Artificial Intelligence (AI) predictive techniques enables auto diagnosis and reduces detection errors compared to exclusive human expertise. In this paper, we have reviewed the current literature for the last 10 years, from January 2009 to December 2019. The study considered eight most frequently used databases, in which a total of 105 articles were found. A detailed analysis of those articles was conducted in order to classify most used AI techniques for medical diagnostic systems. We further discuss various diseases along with corresponding techniques of AI, including Fuzzy Logic, Machine Learning, and Deep Learning. This research paper aims to reveal some important insights into current and previous different AI techniques in the medical field used in today’s medical research, particularly in heart disease prediction, brain disease, prostate, liver disease, and kidney disease. Finally, the paper also provides some avenues for future research on AI-based diagnostics systems based on a set of open problems and challenges.</p>", unsafe_allow_html=True)
   


def __main__():
    with st.sidebar:
        #st.title("Menu")
        selected = option_menu(menu_title = "Menu", 
                            options = ["Home", "Brain Stroke Prediction", "Diabetes Prediction", "Heart Attack Prediction", "Prediction based on Symptoms"], 
                            default_index=0, 
                            menu_icon=None, 
                           icons=None, 
                            orientation="vertical",
                            styles=None, 
                            key=None)
   

    if selected == 'Brain Stroke Prediction':
        stroke_home()
    if selected == 'Diabetes Prediction':
        diabetes_home()
    if selected == 'Heart Attack Prediction':
        heartAttack()
    if selected == 'Home':
        welcome()
    if selected == 'Prediction based on Symptoms':
        symptoms()


__main__()