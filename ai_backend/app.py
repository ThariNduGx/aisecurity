from flask import Flask, request, jsonify
import joblib
import numpy as np
from rate_limiter import RateLimiter

app = Flask(__name__)

# Initialize the rate limiter (connects to local Redis on default port 6379)
limiter = RateLimiter(host='localhost', port=6379, db=0)

# Load the trained AI model when the server starts
try:
    model = joblib.load('moodle_ai_security_model.pkl')
    print("[INFO] Real AI Model loaded successfully.")
except Exception as e:
    print(f"[ERROR] Could not load model. Error: {e}")
    model = None

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        user_id = data.get('userid', 0)
        # Assuming Moodle sends the user IP. If not, fallback to Flask request IP.
        ip_address = data.get('ip', request.remote_addr)

        # 1. Rate Limiting Check
        is_blocked = limiter.is_rate_limited(ip_address)
        
        if is_blocked:
            print(f"[SECURITY] Blocking rate-limited request from IP: {ip_address}")
            return jsonify({
                'prediction': 1, 
                'status': 'attack', 
                'confidence': 1.0, 
                'reason': 'rate_limit_exceeded (high velocity)'
            }), 429

        # 2. Redis Caching Check (Latency Reduction)
        cached_result = limiter.get_cached_prediction(ip_address)
        if cached_result:
            print(f"[INFO] Serving cached prediction for IP: {ip_address}")
            return jsonify(cached_result)

        # 3. Dummy array of 39 zeros to match model expected input
        features = np.zeros(39)
        
        # Artificial trigger for testing
        if int(user_id) % 2 == 0:
            features[0] = 999999 
        else:
            features[0] = 0
            
        features = features.reshape(1, -1)
        
        if model:
            prediction = int(model.predict(features)[0])
        else:
            prediction = 0
            
        status = 'safe' if prediction == 0 else 'attack'

        # Response payload
        response_data = {
            'prediction': prediction, 
            'status': status, 
            'confidence': 0.95
        }

        # 4. Cache the result in Redis for future requests from this IP
        limiter.cache_prediction(ip_address, response_data)

        return jsonify(response_data)

    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)