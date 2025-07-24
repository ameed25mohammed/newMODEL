from flask import Flask, request, jsonify
import numpy as np
import joblib
import os

# تحميل النموذج ومعلومات النموذج
model = joblib.load('drug_addiction_random_forest_model.pkl')
model_info = joblib.load('model_info.pkl')

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Drug Addiction Prediction API is running on Render!',
        'status': 'success',
        'model_type': 'Random Forest (PKL)',
        'model_info': model_info if 'model_info' in globals() else 'Model info not available'
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'input' not in data:
            return jsonify({'error': 'Missing input data'}), 400
            
        input_data = np.array(data['input']).reshape(1, -1)
        
        # التنبؤ باستخدام النموذج
        prediction = model.predict(input_data)[0]
        
        # الحصول على احتمالية التنبؤ إذا كان النموذج يدعمها
        try:
            prediction_prob = model.predict_proba(input_data)[0]
            # إذا كان التصنيف ثنائي، نأخذ احتمالية الفئة الإيجابية
            if len(prediction_prob) == 2:
                probability = float(prediction_prob[1])
            else:
                probability = float(max(prediction_prob))
        except:
            # إذا لم يدعم النموذج predict_proba
            probability = None
        
        result = {
            'prediction': int(prediction),
            'platform': 'Render'
        }
        
        if probability is not None:
            result['probability'] = probability
            
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model-info', methods=['GET'])
def get_model_info():
    try:
        return jsonify({
            'model_info': model_info,
            'model_type': 'Random Forest',
            'status': 'success'
        })
    except Exception as e:
        return jsonify({'error': f'Error retrieving model info: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_info_loaded': 'model_info' in globals() and model_info is not None
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)