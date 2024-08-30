# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 10:14:07 2024

@author: 15694
"""

import numpy as np
import pandas as pd
import pickle
import joblib
import streamlit as st
from sklearn.preprocessing import StandardScaler

# Loading the saved model and scaler
loaded_model = pickle.load(open('C:/Users/15694/Desktop/Cross selling/code/Trained_model.sav', 'rb'))
scaler = joblib.load('C:/Users/15694/Desktop/Cross selling/code/scaler.save')

def Cross_Selling_Loan_Prediction(customer_id, customer_name, input_data):
    input_data_as_numpy_array = np.array(input_data)
    input_data_reshape = input_data_as_numpy_array.reshape(1, -1)
    
    std_data = scaler.transform(input_data_reshape)
    prediction = loaded_model.predict(std_data)
    
    if prediction == 0:
        loan_range = "Loan_Range (0) = LNAMOUNT <= 1 Lakh"
    elif prediction == 1:
        loan_range = "Loan_Range (1) = 1 Lakh < LNAMOUNT <= 5 Lakh"
    elif prediction == 2:
        loan_range = "Loan_Range (2) = 5 Lakh < LNAMOUNT <= 1M"
    elif prediction == 3:
        loan_range = "Loan_Range (3) = 1M < LNAMOUNT <= 5M"
    elif prediction == 4:
        loan_range = "Loan_Range (4) = 5M < LNAMOUNT <= 10M"
    elif prediction == 5:
        loan_range = "Loan_Range (5) = LNAMOUNT > 10M"
    else:
        loan_range = "Invalid prediction"
    
    return customer_id, customer_name, loan_range

def main():
    st.title('Cross Selling Loan Prediction Web App')
    
    st.sidebar.header('Choose Input Method:')
    input_method = st.sidebar.radio('', ('Manual Input', 'Upload Excel'))
    
    if input_method == 'Manual Input':
        st.subheader('Enter Customer Data:')
        TOTAL_AVG_BAL = st.text_input('Total Average Balance')
        SIX_MONTH_BAL_OS_FD = st.text_input('Last six month outstanding balance of FD')
        NPA_FLAG = st.text_input('Non Performing Asset Flag')  
        SIX_MONTH_BAL_OS_LEASING = st.text_input('Last six month outstanding balance of Leasing')
        CUSTOMER_PROFITABILITY = st.text_input('Customer Profitability')
        SIX_MONTH_BAL_OS_SAVINGS = st.text_input('Last six month outstanding balance of Saving')
        
        if st.button('Predict'):
            input_data = [TOTAL_AVG_BAL, SIX_MONTH_BAL_OS_FD, NPA_FLAG, SIX_MONTH_BAL_OS_LEASING, CUSTOMER_PROFITABILITY, SIX_MONTH_BAL_OS_SAVINGS]
            customer_id = 'This'
            customer_name = 'Customer can get = '
            prediction = Cross_Selling_Loan_Prediction(customer_id, customer_name, input_data)
            st.success(prediction)
    
    elif input_method == 'Upload Excel':
        uploaded_file = st.file_uploader("Upload Excel file", type=["xls", "xlsx"])
        
        if uploaded_file is not None:
            input_df = pd.read_excel(uploaded_file)
            
            if 'Customer_ID' not in input_df.columns or 'Customer_Name' not in input_df.columns:
                st.error("Input file must contain 'Customer_ID' and 'Customer_Name' columns.")
            else:
                predictions = []
                for index, row in input_df.iterrows():
                    customer_id = row['Customer_ID']
                    customer_name = row['Customer_Name']
                    input_data = row.drop(['Customer_ID', 'Customer_Name']).tolist()
                    prediction = Cross_Selling_Loan_Prediction(customer_id, customer_name, input_data)
                    predictions.append(prediction)
                
                result_df = pd.DataFrame(predictions, columns=['Customer_ID', 'Customer_Name', 'Prediction'])
                st.write(result_df)

if __name__ == '__main__':
    main()
