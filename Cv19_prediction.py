import streamlit as st
import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Input


model = load_model('my_usa_cv19_model.keras')


st.title("COVID-19 Weekly Prediction APP Using LSTM model")

st.subheader("In Each Input Slider, Enter The Last Three Weeks Data. For Any Information About How To Use This Prediction Tool, Contact Jacob Akubire @ jaakubire@gmail.com")

week1_active = st.number_input('Enter Week 1 - Active Cases: This is Cumulative Cases')
week1_cases = st.number_input('Enter Week 1 - New Cases')
week1_death = st.number_input('Enter Week 1 - New Deaths')


week2_active = st.number_input('Enter Week 2 - Active Cases: This is Cumulative Cases')
week2_cases = st.number_input('Enter Week 2 - New Cases')
week2_death = st.number_input('Enter Week 2 - New Deaths')

week3_active = st.number_input('Enter Week 3 - Active Cases: This is Cumulative Cases')
week3_cases = st.number_input('Enter Week 3 - New Cases')
week3_death = st.number_input('Enter Week 3 - New Deaths')


if st.button('Predict'):
    input_data = np.array([
        [week1_active, week1_cases, week1_death],
        [week2_active, week2_cases, week2_death],
        [week3_active, week3_cases, week3_death]
                           ]
        
    )

    input_data[input_data==0]=1
    
    input_data_log = np.log(input_data)

    input_data_log = input_data_log.reshape(1, 3,3)

    prediction = model.predict(input_data_log)
    prediction = np.exp(prediction)

    st.subheader('Prediction for next week')

    st.write(f'Predicted Active Cases: {prediction[0][0]}')
    st.write(f'Predicted New Cases: {prediction[0][1]}')
    st.write(f'Predicted New Deaths: {prediction[0][2]}')
