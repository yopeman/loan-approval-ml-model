from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load trained model and preprocessors
model = joblib.load('loan_model.joblib')
scaler = joblib.load('scaler.joblib')
label_encoders = joblib.load('label_encoders.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Accept both JSON and form data
        if request.is_json:
            request_data = request.get_json()
        else:
            request_data = request.form
        
        data = {
            'loan_id': 0,
            'no_of_dependents': int(request_data.get('no_of_dependents')),
            'education': request_data.get('education'),
            'self_employed': request_data.get('self_employed'),
            'income_annum': int(request_data.get('income_annum')),
            'loan_amount': int(request_data.get('loan_amount')),
            'loan_term': int(request_data.get('loan_term')),
            'cibil_score': int(request_data.get('cibil_score')),
            'residential_assets_value': int(request_data.get('residential_assets_value')),
            'commercial_assets_value': int(request_data.get('commercial_assets_value')),
            'luxury_assets_value': int(request_data.get('luxury_assets_value')),
            'bank_asset_value': int(request_data.get('bank_asset_value'))
        }
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Encode categorical features
        for col in ['education', 'self_employed']:
            df[col] = label_encoders[col].transform(df[col])
        
        # Scale features
        scaled_features = scaler.transform(df)
        
        # Make prediction
        prediction = model.predict(scaled_features)[0]
        probability = model.predict_proba(scaled_features)[0]
        
        result = "Approved ✅" if prediction == 0 else "Rejected ❌"
        status = "approved" if prediction == 0 else "rejected"
        confidence = round(probability[prediction] * 100, 2)
        
        return jsonify({
            'success': True,
            'result': result,
            'status': status,
            'confidence': confidence
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)