import streamlit as st
import numpy as np
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Input


model = load_model('my_usa_cv19_model.keras')


st.title("COVID-19 Weekly Prediction Using LSTM model")

st.subheader("Enter the last three weeks data")

week1_active = st.number_input('Week 1 - Active cases: this is cumulative cases')
week1_cases = st.number_input('week 1 - New cases')
week1_death = st.number_input('week 1 - New deaths')


week2_active = st.number_input('Week 2 - Active cases: this is cumulative cases')
week2_cases = st.number_input('week 2 - New cases')
week2_death = st.number_input('week 2 - New deaths')

week3_active = st.number_input('Week 3 - Active cases: this is cumulative cases')
week3_cases = st.number_input('week 3 - New cases')
week3_death = st.number_input('week 3 - New deaths')


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

    predition = model.predict(input_data_log)
    prediction = np.exp(prediction)

    st.subheader('Prediction for next week')

    st.write(f'Predicted Active Cases: {prediction[0][0]}')
    st.write(f'Predicted New Cases: {prediction[0][1]}')
    st.write(f'Predicted New Deaths: {prediction[0][2]}')
