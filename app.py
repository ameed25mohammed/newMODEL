from flask import Flask, request, jsonify
import numpy as np
import joblib
import os

# تحميل النموذج
try:
    model = joblib.load('drug_addiction_random_forest_model.pkl')
except:
    model = None

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'API is running',
        'status': 'success'
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'error': 'Model not available'}), 500
            
        data = request.get_json()
        if not data or 'input' not in data:
            return jsonify({'error': 'Invalid input'}), 400
        
        input_values = data['input']
        
        if len(input_values) != 27:
            return jsonify({'error': 'Wrong number of features'}), 400

        input_array = np.array(input_values).reshape(1, -1)
        prediction = model.predict(input_array)[0]
        
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
    
    except:
        return jsonify({'error': 'Prediction failed'}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
