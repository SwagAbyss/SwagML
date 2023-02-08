#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 05:01:25 2023

@author: swag
"""

import numpy as np
import pickle
import streamlit as st

# loading the saved model
loaded_model = pickle.load(open('/Users/swag/Documents/GitHub/SwagML/Projects/diabetes/trained_model.sav', 'rb'))

# dcreating a fuction for prediction
def diabetes_prediction(input_data):
    
    # changing the input data into numpy array
    input_data_as_array = np.asarray(input_data)

    # reshaping the array as we are predicting for one instance
    input_data_reshape = input_data_as_array.reshape(1, -1)

    # predicting the label
    prediction = loaded_model.predict(input_data_reshape)
    # print(prediction)

    if prediction == 0:
        return'The person is not Diabetic'
    else:
        return'The person is Diabetic'

def main():

    # Giving a title
    st.title('Diabetes Prediction App')

    # Getting the input from the user
    Pregnancies = st.text_input('Number of Pregnencies')
    Glucose = st.text_input('Glucose value')
    BloodPressure = st.text_input('Blood Pressure Level')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the Person')

    # code for Prediction
    diagnosis = ''

    # creating a button for prediction
    if st.button('Submit'):
        diagnosis = diabetes_prediction(
            [Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]
        )

    st.success(diagnosis)

if __name__=='__main__':
    main()