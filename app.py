from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os

# تحميل النموذج
model = joblib.load('drug_addiction_random_forest_model.pkl')

# تعريف أسماء الأعمدة بنفس ترتيب التدريب
feature_names = [ 
    'Education', 'Family relationship', 'Enjoyable with', 'Addicted person in family',
    'no. of friends', 'Withdrawal symptoms', "friends' houses at night",
    'Living with drug user', 'Smoking', 'Friends influence', 'If chance given to taste drugs',
    'Easy to control use of drug', 'Frequency of drug usage', 'Gender', 'Conflict with law',
    'Failure in life', 'Suicidal thoughts', 'Satisfied with workplace', 'Case in court',
    'Ever taken drug',
    'Mental_emotional problem_Angry', 'Mental_emotional problem_Depression',
    'Mental_emotional problem_Stable', 'Motive about drug_Curiosity',
    'Motive about drug_Enjoyment', 'Live with_Alone', 'Spend most time_Friends'
]

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Drug Addiction Prediction API is running on Render!',
        'status': 'success',
        'model_type': 'Random Forest (PKL)',
        'total_features': len(feature_names),
        'features': feature_names
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # التحقق من وجود البيانات
        data = request.get_json()
        if not data or 'input' not in data:
            return jsonify({'error': 'Missing input data. Please provide data in format: {"input": [values]}'}), 400
        
        input_values = data['input']
        
        # التحقق من عدد المتغيرات
        if len(input_values) != len(feature_names):
            return jsonify({
                'error': f'Expected {len(feature_names)} features, but got {len(input_values)}',
                'expected_features': len(feature_names),
                'received_features': len(input_values),
                'feature_names': feature_names
            }), 400

        # تحويل البيانات إلى DataFrame بالأعمدة المسماة
        input_df = pd.DataFrame([input_values], columns=feature_names)
        
        # التنبؤ
        prediction = model.predict(input_df)[0]
        
        # احتمالية التنبؤ (إن وجدت)
        probability = None
        try:
            prediction_prob = model.predict_proba(input_df)[0]
            if len(prediction_prob) == 2:
                probability = float(prediction_prob[1])  # احتمالية الفئة الإيجابية
            else:
                probability = float(max(prediction_prob))
        except Exception as prob_error:
            print(f"Probability calculation error: {prob_error}")
        
        result = {
            'prediction': int(prediction),
            'prediction_label': 'High Risk' if int(prediction) == 1 else 'Low Risk',
            'platform': 'Render',
            'model_type': 'Random Forest'
        }

        if probability is not None:
            result['probability'] = probability
            result['confidence'] = f"{probability:.2%}"

        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'error_type': type(e).__name__,
            'message': 'An error occurred during prediction'
        }), 500

@app.route('/model-info', methods=['GET'])
def get_model_info():
    return jsonify({
        'model_type': 'Random Forest',
        'total_features': len(feature_names),
        'feature_names': feature_names,
        'status': 'success',
        'description': 'Drug Addiction Prediction Model'
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'feature_count': len(feature_names),
        'api_version': '1.0'
    })

@app.route('/test-input', methods=['GET'])
def test_input():
    """إنشاء مثال على البيانات المطلوبة للاختبار"""
    # مثال على بيانات وهمية للاختبار
    sample_input = [0] * len(feature_names)  # كل القيم = 0 كمثال
    
    return jsonify({
        'sample_input': {
            'input': sample_input
        },
        'feature_count': len(feature_names),
        'feature_names': feature_names,
        'usage': 'Use this sample input format for testing the /predict endpoint'
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=True)
