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
        # Cache is keyed by IP + userid so different users are not mixed up
        cache_key_id = f"{ip_address}:{user_id}"
        cached_result = limiter.get_cached_prediction(cache_key_id)
        if cached_result:
            print(f"[INFO] Serving cached prediction for IP: {ip_address}, user: {user_id}")
            return jsonify(cached_result)

        # 3. Create dummy feature array based on model expected features
        feature_count = model.n_features_in_ if model else 39
        features = np.zeros(feature_count)

        # Artificial trigger for testing:
        # Even userid (e.g. admin=2) → sets dur=999999 → model sees anomaly
        # Odd userid  (e.g. student=1) → dur=0 → model sees normal traffic
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

        print(f"[INFO] user_id={user_id}, ip={ip_address}, "
              f"features[0]={features[0][0]}, prediction={prediction}, status={status}")

        # Response payload
        response_data = {
            'prediction': prediction,
            'status': status,
            'confidence': 0.95
        }

        # 4. Cache the result in Redis for future requests from this IP+user
        limiter.cache_prediction(cache_key_id, response_data)

        return jsonify(response_data)

    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
