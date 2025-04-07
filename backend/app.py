# backend/app.py

from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = joblib.load('../model/sentiment_model.pkl')

@app.route('/')
def index():
    return "API is live!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text')
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        pred = model.predict([text])[0]
        result = 'Positive' if pred == 1 else 'Negative'
        return jsonify({'sentiment': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
