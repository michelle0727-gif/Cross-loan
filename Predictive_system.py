# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import pickle
import joblib   

# Loading the saved model
loaded_model = pickle.load(open('C:/Users/15694/Desktop/Cross selling/code/Trained_model.sav', 'rb'))

input_data = (11434729.22, 4236559.14, 0, 0.0, 45987.69, 417459.07)

data = {
    'TOTAL_AVG_BAL': [11434729.22],
    'SIX_MONTH_BAL_OS_FD': [4236559.14],
    'NPA_FLAG': [0],
    'SIX_MONTH_BAL_OS_LEASING': [0.0],
    'CUSTOMER_PROFITABILITY': [45987.69],
    'SIX_MONTH_BAL_OS_SAVINGS': [417459.07]
}

df_val = pd.DataFrame(data)

# Assuming input_data_as_numpy_array is defined somewhere in your code
input_data_as_numpy_array = np.array(input_data)
input_data_reshape = input_data_as_numpy_array.reshape(1, -1)
print(input_data_reshape)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
std_data = scaler.fit_transform(input_data_reshape)

# Load scaler from file
sc = joblib.load('C:/Users/15694/Desktop/Cross selling/code/scaler.save')
df_val1 = sc.transform(df_val)

prediction = loaded_model.predict(df_val1)
print(prediction)

if prediction == 0:
    print("Loan_Range = LNAMOUNT <= 1 Lakh")
elif prediction == 1:
    print("Loan_Range = 1 Lakh < LNAMOUNT <= 5 Lakh")
elif prediction == 2:
    print("Loan_Range = 5 Lakh < LNAMOUNT <= 1M")
elif prediction == 3:
    print("Loan_Range = 1M < LNAMOUNT <= 5M")
elif prediction == 4:
    print("Loan_Range = 5M < LNAMOUNT <= 10M")
elif prediction == 5:
    print("Loan_Range = LNAMOUNT > 10M")
else:
    print("Invalid prediction")


