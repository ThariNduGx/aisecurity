from flask import Flask, request, jsonify
import joblib
import numpy as np
from rate_limiter import RateLimiter

app = Flask(__name__)

# Initialize the rate limiter (connects to local Redis on default port 6379)
limiter = RateLimiter(host='localhost', port=6379, db=0)

# Load the trained multi-class model (Decision Tree, tuned max_depth)
try:
    model = joblib.load('moodle_ai_security_model.pkl')
    print("[INFO] AI model loaded successfully.")
except Exception as e:
    print(f"[ERROR] Could not load model: {e}")
    model = None

# Load the StandardScaler fitted during training.
# Live features MUST be scaled with the same scaler before prediction.
try:
    scaler = joblib.load('scaler.pkl')
    print("[INFO] Scaler loaded successfully.")
except Exception as e:
    print(f"[WARNING] scaler.pkl not found: {e}  (features will not be scaled)")
    scaler = None


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
                'prediction' : 'RateLimit',
                'status'     : 'attack',
                'confidence' : 1.0,
                'reason'     : 'rate_limit_exceeded',
            }), 429

        # 2. Redis Cache (keyed by IP + userid for per-user isolation)
        cache_key     = f"{ip_address}:{user_id}"
        cached_result = limiter.get_cached_prediction(cache_key)
        if cached_result:
            print(f"[INFO] Cache hit for {cache_key}")
            return jsonify(cached_result)

        # 3. Build feature vector from the request payload.
        #    simulate_attack.py sends a 'features' array with all numeric
        #    UNSW-NB15 columns in order.  The Moodle PHP plugin sends
        #    userid/ip only; those requests fall back to the zero vector
        #    (safe baseline) until full telemetry is wired up.
        feature_count = model.n_features_in_ if model else 39

        raw_features = data.get('features')
        if raw_features and len(raw_features) == feature_count:
            features = np.array(raw_features, dtype=float)
        else:
            features = np.zeros(feature_count)

        features = features.reshape(1, -1)

        # 4. Scale with the same StandardScaler used during training
        if scaler is not None:
            features = scaler.transform(features)

        # 5. Predict – model returns attack_cat string directly (e.g. "DoS")
        if model:
            attack_cat = str(model.predict(features)[0])
        else:
            attack_cat = 'Normal'

        status = 'safe' if attack_cat == 'Normal' else 'attack'

        print(f"[INFO] ip={ip_address} user={user_id} "
              f"attack_cat={attack_cat} status={status}")

        response_data = {
            'prediction' : attack_cat,
            'status'     : status,
            'confidence' : 0.95,
        }

        # 6. Cache the result for future identical requests
        limiter.cache_prediction(cache_key, response_data)

        return jsonify(response_data)

    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
