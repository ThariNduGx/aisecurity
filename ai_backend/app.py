from flask import Flask, request, jsonify
import joblib
import numpy as np
from rate_limiter import RateLimiter

app = Flask(__name__)

# Initialize the rate limiter (connects to local Redis on default port 6379)
limiter = RateLimiter(host='localhost', port=6379, db=0)

# Load the trained multi-class AI model
try:
    model = joblib.load('moodle_ai_security_model.pkl')
    print("[INFO] AI model loaded successfully.")
except Exception as e:
    print(f"[ERROR] Could not load model: {e}")
    model = None

# Load the label encoder so integer predictions map back to attack_cat strings
try:
    label_encoder = joblib.load('label_encoder.pkl')
    print(f"[INFO] Label encoder loaded. Classes: {list(label_encoder.classes_)}")
except Exception as e:
    print(f"[WARNING] Label encoder not found: {e}  (binary fallback active)")
    label_encoder = None


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        user_id    = data.get('userid', 0)
        ip_address = data.get('ip', request.remote_addr)

        # 1. Rate Limiting Check
        if limiter.is_rate_limited(ip_address):
            print(f"[SECURITY] Blocking rate-limited IP: {ip_address}")
            return jsonify({
                'prediction' : 1,
                'attack_cat' : 'DoS',
                'status'     : 'attack',
                'confidence' : 1.0,
                'reason'     : 'rate_limit_exceeded'
            }), 429

        # 2. Redis Caching (keyed by IP + userid for per-user isolation)
        cache_key_id  = f"{ip_address}:{user_id}"
        cached_result = limiter.get_cached_prediction(cache_key_id)
        if cached_result:
            print(f"[INFO] Cache hit for {cache_key_id}")
            return jsonify(cached_result)

        # 3. Build feature vector from the request payload.
        #    The simulation tool (simulate_attack.py) sends a 'features' array
        #    containing the 39 numeric UNSW-NB15 columns in the correct order.
        #    The Moodle PHP plugin sends whatever network metadata it can collect.
        feature_count = model.n_features_in_ if model else 39

        raw_features = data.get('features')
        if raw_features and len(raw_features) == feature_count:
            # Real network features supplied by the caller
            features = np.array(raw_features, dtype=float)
        else:
            # Fallback: zero vector (treated as normal baseline traffic).
            # This path is hit when Moodle sends only userid/ip without
            # full network telemetry. No artificial attack triggering here.
            features = np.zeros(feature_count)

        features = features.reshape(1, -1)

        # 4. Run inference
        if model:
            prediction = int(model.predict(features)[0])
        else:
            prediction = 0

        # 5. Decode integer prediction → attack category string
        if label_encoder:
            attack_cat = label_encoder.inverse_transform([prediction])[0]
        else:
            attack_cat = 'Unknown'

        status = 'safe' if attack_cat == 'Normal' else 'attack'

        print(f"[INFO] ip={ip_address} user={user_id} "
              f"prediction={prediction} attack_cat={attack_cat} status={status}")

        response_data = {
            'prediction' : prediction,
            'attack_cat' : attack_cat,
            'status'     : status,
            'confidence' : 0.95,
        }

        # 6. Cache for future identical requests
        limiter.cache_prediction(cache_key_id, response_data)

        return jsonify(response_data)

    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
