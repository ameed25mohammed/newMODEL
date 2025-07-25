from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import os

# تحميل النموذج
try:
    model = joblib.load('drug_addiction_random_forest_model.pkl')
except:
    model = None

app = Flask(__name__)
CORS(app)  # إضافة CORS

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'API is running',
        'status': 'success'
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # فحص النموذج
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
            
        # فحص البيانات
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data'}), 400
            
        if 'input' not in data:
            return jsonify({'error': 'Missing input field'}), 400
        
        input_values = data['input']
        
        # فحص عدد المتغيرات
        if len(input_values) != 27:
            return jsonify({
                'error': f'Expected 27 features, got {len(input_values)}'
            }), 400

        # تحويل البيانات
        try:
            input_array = np.array(input_values, dtype=float).reshape(1, -1)
        except Exception as e:
            return jsonify({'error': 'Invalid data format'}), 400
        
        # التنبؤ
        prediction = model.predict(input_array)[0]
        
        # الاحتمالية
        try:
            probability = model.predict_proba(input_array)[0][1]
        except:
            probability = None
        
        result = {
            'prediction': int(prediction),
            'status': 'success'
        }
        
        if probability is not None:
            result['probability'] = round(float(probability), 4)
            
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'error': 'Prediction failed',
            'details': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
