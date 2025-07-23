from flask import Flask, request, jsonify, render_template, url_for
import numpy as np
import joblib  # For loading the model and scaler

app = Flask(__name__, static_folder='static', template_folder='templates')


# Load the trained model and scaler
xgb_model = joblib.load('xgboost_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/service')
def service():
    return render_template('service.html')

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/price')
def price():
    return render_template('price.html')

@app.route('/testimonial')
def testimonial():
    return render_template('testimonial.html')

# Define pregnancy-specific normal ranges
pregnancy_normal_ranges = {
    "Age": (18, 35),
    "Body Temperature (F)": (97, 99),
    "Heart Rate (bpm)": (70, 110),
    "Systolic Blood Pressure (mm Hg)": (110, 120),
    "Diastolic Blood Pressure (mm Hg)": (70, 80),
    "BMI (kg/m²)": (18.5, 24.9),
    "Blood Glucose (HbA1c)": (0, 42),  # Upper limit for HbA1c
    "Blood Glucose (Fasting hour-mg/dL)": (3.3, 5.1),
}

# Define risk level function
def risk_level(outcome):
    if outcome == 0:
        return "Low Risk"
    elif outcome == 1:
        return "Medium Risk"
    elif outcome == 2:
        return "High Risk"
    else:
        return "Unknown Risk"

# Function to explain risk factors
def explain_risk_factors(user_input):
    explanations = []
    for feature, value in user_input.items():
        if feature in pregnancy_normal_ranges:
            low, high = pregnancy_normal_ranges[feature]
            if value < low:
                explanations.append(f"Low {feature} ({value}) raises the risk during pregnancy.")
            elif value > high:
                explanations.append(f"High {feature} ({value}) raises the risk during pregnancy.")
    return explanations

# Flask route to handle prediction
@app.route('/check', methods=['GET', 'POST'])
def check():
    result = None
    explanations = None
    
    if request.method == 'POST':
        # Get the data from the form
        age = float(request.form['Age'])
        height = float(request.form['height'])  # Height in cm
        weight = float(request.form['weight'])  # Weight in kg
        body_temp = float(request.form['Body_Temperature'])
        heart_rate = float(request.form['Heart_Rate'])
        systolic_bp = float(request.form['Systolic_BP'])
        diastolic_bp = float(request.form['Diastolic_BP'])
        blood_glucose_hba1c = float(request.form['Blood_Glucose_HbA1c'])
        blood_glucose_fasting = float(request.form['Blood_Glucose_Fasting'])

        # Calculate BMI
        bmi = weight / ((height / 100) ** 2)

        # Prepare input data for model
        user_input = {
            "Age": age,
            "Body Temperature (F)": body_temp,
            "Heart Rate (bpm)": heart_rate,
            "Systolic Blood Pressure (mm Hg)": systolic_bp,
            "Diastolic Blood Pressure (mm Hg)": diastolic_bp,
            "BMI (kg/m²)": bmi,
            "Blood Glucose (HbA1c)": blood_glucose_hba1c,
            "Blood Glucose (Fasting hour-mg/dl)": blood_glucose_fasting
        }

        feature_order = [
            "Age", "Body Temperature (F)", "Heart Rate (bpm)", 
            "Systolic Blood Pressure (mm Hg)", "Diastolic Blood Pressure (mm Hg)",
            "BMI (kg/m²)", "Blood Glucose (HbA1c)", "Blood Glucose (Fasting hour-mg/dl)"
        ]

        input_array = [[user_input[feature] for feature in feature_order]]
        input_scaled = scaler.transform(input_array)

        # Predict the outcome using the model
        predicted_outcome = xgb_model.predict(input_scaled)[0]

        # Get the risk level
        risk = risk_level(predicted_outcome)

        # Explain the risk factors
        explanations = explain_risk_factors(user_input)

        # Prepare result
        result = {
            "prediction": risk,
            "explanations": explanations
        }

    return render_template('check.html', result=result)



# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
