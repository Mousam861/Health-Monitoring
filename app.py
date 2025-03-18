import os
from flask import Flask, render_template, request, jsonify
import numpy as np
import sklearn
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/bmi')
def bmi():
    return render_template('bmi.html')

# Load the diabetes model and scaler
try:
    dia_model = pickle.load(open('model.pkl', 'rb'))
    dia_scaler = pickle.load(open('scaler.pkl', 'rb')) # Load saved scaler instead of creating new one
except FileNotFoundError:
    print("Error: Required model files not found")

@app.route('/diabetes')
def diabetes():
    return render_template('dia.html')

@app.route('/predictdia', methods=['POST'])
def predictdia():
    try:
        # Get values from form
        features = []
        features.append(float(request.form['Glucose Level']))
        features.append(float(request.form['Insulin']))
        features.append(float(request.form['BMI']))
        features.append(float(request.form['Age']))
        
        # Convert to numpy array and reshape
        final_features = np.array(features).reshape(1, -1)
        
        # Scale the features using pre-fitted scaler
        final_features_scaled = dia_scaler.transform(final_features)
        
        # Make prediction
        prediction = dia_model.predict(final_features_scaled)
        
        output = prediction[0]
        
        if output == 1:
            pred = "You have Diabetes, please consult a Doctor."
        else:
            pred = "You don't have Diabetes."
            
        return render_template('resultdia.html', prediction_text=pred)
    except Exception as e:
        return render_template('resultdia.html', prediction_text="Error making prediction")

@app.route('/predictAction', methods=['POST'])
def predictAction():
    try:
        if request.method == 'POST':
            name = request.form['name']
            age = float(request.form['age'])
            maritalstatus = request.form['maritalstatus']
            worktype = request.form['Worktype']
            residence = request.form['Residence']
            gender = request.form['gender']
            bmi = float(request.form['bmi'])
            gluclevel = float(request.form['gluclevel'])
            smoke = request.form['Smoke']
            hypertension = request.form['Hypertension']
            heartdisease = request.form['Heartdisease']

            res = {'urban': 1, 'rural': 0}
            gen = {'Female': 0, 'Male': 1}
            msts = {'married': 1, 'not married': 0}
            wktype = {'privatejob': 2, 'govtemp': 1, 'selfemp': 3}
            smke = {'formerly-smoked': 1, 'non-smoker': 2, 'smoker': 3}
            hypten = {'hypten': 1, 'nohypten': 0}
            hrtdis = {'heartdis': 1, 'noheartdis': 0}

            residence = res[residence]
            gender = gen[gender]
            maritalstatus = msts[maritalstatus]
            worktype = wktype[worktype]
            smoke = smke[smoke]
            hypertension = hypten[hypertension]
            heartdisease = hrtdis[heartdisease]

            # Load model once
            try:
                model = pickle.load(open('strokenew.pkl', 'rb'))
            except FileNotFoundError:
                return render_template('result.html', a="Error: Model file not found")

            array = [[gender, age, hypertension, heartdisease, maritalstatus, worktype, residence, gluclevel, bmi, smoke]]
            array = np.array(array, dtype='float64')
            
            pred_stroke = model.predict(array)
            result = int(pred_stroke[0])
            
            if result == 0:
                message = f"{name}, you have a less chance of getting stroke 😀"
            else:
                message = f"{name}, you have a high chance of getting stroke 😔"
                
            return render_template('result.html', a=message)
    except Exception as e:
        return render_template('result.html', a=f"Error processing request: {str(e)}")

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/counsel')
def counsel():
    return render_template('counsel.html')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
