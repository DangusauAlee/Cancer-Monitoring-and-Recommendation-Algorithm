# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 00:04:13 2023

@author: SPYDER
"""
from flask import Flask, render_template, request
import joblib
import numpy as np
import datetime

app = Flask(__name__)

# Load the model, encoder, and scaler
model = joblib.load('cancer_model.pkl')
encoder = joblib.load('encoders.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get user inputs from the form
    Tumor_Size = float(request.form['Tumor_Size'])
    Reginol_Node_Positive = float(request.form['Reginol_Node_Positive'])
    Stage_6 = request.form['Stage_6']
    N_Stage = request.form['N_Stage']
    A_Stage = request.form['A_Stage']
    Survival_Months = float(request.form['Survival_Months'])
    T_Stage = request.form['T_Stage']

    # Encode categorical variables
    Stage_6 = encoder.transform([Stage_6])
    N_Stage = encoder.transform([N_Stage])
    A_Stage = encoder.transform([A_Stage])
    T_Stage = encoder.transform([T_Stage])

    # Combine encoded categorical variables
    categorical_features = np.concatenate((Stage_6, N_Stage, A_Stage, T_Stage), axis=1)

    # Combine all features
    features = np.array([[Tumor_Size, Reginol_Node_Positive, Survival_Months]])
    features = np.concatenate((features, categorical_features), axis=1)

    # Scale features
    scaled_features = scaler.transform(features)

    # Make prediction
    prediction = model.predict(scaled_features)

    priority = "High Priority" if prediction == 0 else "Low Priority"

    # Create a report
     
    report = f"Tumor Size: {Tumor_Size}\nRegional Node Positive: {Reginol_Node_Positive}\n" \
             f"6th Stage: {Stage_6}\nN Stage: {N_Stage}\nA Stage: {A_Stage}\n" \
             f"Survival Months: {Survival_Months}\nT Stage: {T_Stage}\n\n" \
             f"Priority: {priority}\n"

    return render_template('index.html', prediction_text=report)
if __name__ == '__main__':
    app.run(debug=False)
