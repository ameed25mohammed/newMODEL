from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import joblib
import os

# تحميل النموذج
try:
    model = joblib.load('drug_addiction_random_forest_model.pkl')
except:
    model = None

# أسماء الأعمدة الصحيحة حسب النموذج
feature_names = [
    'Education', 'Family relationship', 'Enjoyable with', 'Addicted person in family',
    'no. of friends', 'Withdrawal symptoms', "friends' houses at night",
    'Living with drug user', 'Smoking', 'Friends influence', 'If chance given to taste drugs',
    'Easy to control use of drug', 'Frequency of drug usage', 'Gender', 'Conflict with law',
    'Failure in life', 'Suicidal thoughts', 'Satisfied with workplace', 'Case in court',
    'Ever taken drug', 'Financials of family',
    'Mental_emotional problem_Angry', 'Mental_emotional problem_Depression',
    'Mental_emotional problem_Stable', 'Motive about drug_Curiosity',
    'Motive about drug_Enjoyment', 
    'Live_with_Alone', 'Live_with_Hostel/Hall', 'Live_with_With Family/Relatives',
    'Spend_most_time_Alone', 'Spend_most_time_Business', 'Spend_most_time_Family/ Relatives',
    'Spend_most_time_Friends', 'Spend_most_time_Hostel', 'Spend_most_time_Job/Work place'
]

app = Flask(__name__)

# إضافة CORS headers لكل الاستجابات
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

# معالجة طلبات OPTIONS
@app.route('/predict', methods=['OPTIONS'])
@app.route('/health', methods=['OPTIONS'])
@app.route('/', methods=['OPTIONS'])
def handle_options():
    response = jsonify({'status': 'ok'})
    return response

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Drug Addiction Prediction API v2.0',
        'status': 'running',
        'model_loaded': model is not None,
        'total_features': len(feature_names),
        'cors_enabled': True
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy' if model is not None else 'unhealthy',
        'model_loaded': model is not None,
        'feature_count': len(feature_names),
        'api_version': '2.0',
        'cors_enabled': True
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # التحقق من تحميل النموذج
        if model is None:
            return jsonify({
                'error': 'Model not loaded',
                'status': 'error'
            }), 500
            
        # التحقق من وجود البيانات
        data = request.get_json()
        if not data:
            return jsonify({
                'error': 'No JSON data provided',
                'status': 'error'
            }), 400
            
        if 'input' not in data:
            return jsonify({
                'error': 'Missing "input" field',
                'status': 'error'
            }), 400
        
        input_values = data['input']
        
        # التحقق من عدد المتغيرات
        if len(input_values) != len(feature_names):
            return jsonify({
                'error': f'Expected {len(feature_names)} features, got {len(input_values)}',
                'expected': len(feature_names),
                'received': len(input_values),
                'status': 'error'
            }), 400

        # تحويل البيانات إلى DataFrame
        try:
            input_df = pd.DataFrame([input_values], columns=feature_names)
        except Exception as e:
            return jsonify({
                'error': 'Invalid data format',
                'details': str(e),
                'status': 'error'
            }), 400
        
        # التنبؤ
        prediction = model.predict(input_df)[0]
        
        # احتمالية التنبؤ
        probability = None
        try:
            prediction_prob = model.predict_proba(input_df)[0]
            if len(prediction_prob) == 2:
                probability = float(prediction_prob[1])
            else:
                probability = float(max(prediction_prob))
        except Exception as prob_error:
            print(f"Probability calculation error: {prob_error}")
        
        result = {
            'prediction': int(prediction),
            'prediction_label': 'High Risk' if int(prediction) == 1 else 'Low Risk',
            'status': 'success',
            'model_type': 'Random Forest',
            'cors_enabled': True
        }

        if probability is not None:
            result.update({
                'probability': round(probability, 4),
                'confidence': f"{probability:.2%}"
            })

        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'error_type': type(e).__name__,
            'status': 'error'
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
