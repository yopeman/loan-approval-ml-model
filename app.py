from flask import Flask, render_template, request, jsonify
from helper import predict_loan

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form

        prediction, probability = predict_loan(data)
        result = "Approved" if prediction == 0 else "Rejected"
        status = result.lower()

        confidence = round(float(probability[prediction]) * 100, 2)
        
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