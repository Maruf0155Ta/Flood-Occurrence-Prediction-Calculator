from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

# Load trained model, scaler, and feature list
model = joblib.load('flood_model.pkl')
scaler = joblib.load('scaler.pkl')
model_features = joblib.load('model_features.pkl')

# Original numeric column names (matching training)
scaler_cols = ['Max_Temp','Min_Temp','Rainfall log','Relative_Humidity',
               'Wind_Speed log','Cloud_Coverage','Bright_Sunshine',
               'X_COR','Y_COR','LATITUDE','LONGITUDE','ALT log','Period']

app = Flask(__name__)

@app.route('/')
def home():
    stations = [col.replace('Station_Names_', '') for col in model_features if col.startswith('Station_Names_')]
    return render_template('index.html', stations=stations)

@app.route('/predict', methods=['POST'])
def predict():
    stations = [col.replace('Station_Names_', '') for col in model_features if col.startswith('Station_Names_')]

    # Step 1: Build full DataFrame with all model features, all zeros
    input_df = pd.DataFrame(0.0, index=[0], columns=model_features)

    # Step 2: Fill numeric values (map HTML names → training names)
    input_df['Max_Temp']          = float(request.form['Max_Temp'])
    input_df['Min_Temp']          = float(request.form['Min_Temp'])
    input_df['Rainfall log']      = float(request.form['Rainfall'])
    input_df['Relative_Humidity'] = float(request.form['Relative_Humidity'])
    input_df['Wind_Speed log']    = float(request.form['Wind_Speed'])
    input_df['Cloud_Coverage']    = float(request.form['Cloud_Coverage'])
    input_df['Bright_Sunshine']   = float(request.form['Bright_Sunshine'])
    input_df['X_COR']             = float(request.form['X_COR'])
    input_df['Y_COR']             = float(request.form['Y_COR'])
    input_df['LATITUDE']          = float(request.form['LATITUDE'])
    input_df['LONGITUDE']         = float(request.form['LONGITUDE'])
    input_df['ALT log']           = float(request.form['ALT'])
    input_df['Period']            = float(request.form['Period'])

    # Step 3: Set selected station to 1
    selected_station = request.form['Station_Name']
    station_col = 'Station_Names_' + selected_station
    if station_col in input_df.columns:
        input_df[station_col] = 1.0

    # Step 4: Scale the ENTIRE row (all 44 columns) — this is how it was trained
    input_scaled = scaler.transform(input_df)
    input_scaled_df = pd.DataFrame(input_scaled, columns=model_features)

    # Step 5: Predict
    flood_prob = model.predict_proba(input_scaled_df)[:, 1][0]
    flood_pred = model.predict(input_scaled_df)[0]
    result = "🚨 Flood Likely" if flood_pred == 1 else "✅ No Flood Risk"

    return render_template(
        'index.html',
        stations=stations,
        prediction=result,
        prob=round(flood_prob, 3),
        values=request.form
    )

if __name__ == "__main__":
    app.run(debug=True)