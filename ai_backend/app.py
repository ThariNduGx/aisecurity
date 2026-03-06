from flask import Flask, request, jsonify
import joblib
import numpy as np
import datetime
from rate_limiter import RateLimiter

app = Flask(__name__)

# Initialize the rate limiter (connects to local Redis on default port 6379)
limiter = RateLimiter(host='localhost', port=6379, db=0)

# Load the trained AI model when the server starts
try:
    model = joblib.load('moodle_app_security_model.pkl')
    print("[INFO] Real AI Model loaded successfully.")
except Exception as e:
    print(f"[ERROR] Could not load model. Error: {e}")
    model = None

def get_login_features(ip_address, user_id):
    """
    Track and return the 4 features the model was trained on:
      - login_attempts_last_5m: how many login calls from this IP in the last 5 minutes
      - distinct_users_from_ip_last_5m: how many unique user IDs from this IP in 5 min
      - hour_of_day: current hour (0-23)
      - day_of_week: current weekday (0=Monday, 6=Sunday)
    Counts are stored in Redis with a 5-minute (300s) TTL.
    """
    try:
        redis = limiter.redis_client
        WINDOW = 300  # 5 minutes in seconds

        # Increment login attempt counter for this IP
        attempts_key = f"login_attempts:{ip_address}"
        login_attempts = redis.incr(attempts_key)
        if login_attempts == 1:
            redis.expire(attempts_key, WINDOW)

        # Add this user to the distinct-users set for this IP
        users_key = f"distinct_users:{ip_address}"
        redis.sadd(users_key, str(user_id))
        redis.expire(users_key, WINDOW)
        distinct_users = redis.scard(users_key)

    except Exception:
        # If Redis is unavailable, fall back to neutral values
        login_attempts = 1
        distinct_users = 1

    now = datetime.datetime.now()
    hour_of_day = now.hour
    day_of_week = now.weekday()  # 0=Monday, 6=Sunday

    return login_attempts, distinct_users, hour_of_day, day_of_week


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        user_id = data.get('userid', 0)
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

        # 3. Extract real features and run the trained model
        login_attempts, distinct_users, hour_of_day, day_of_week = get_login_features(ip_address, user_id)
        features = np.array([[login_attempts, distinct_users, hour_of_day, day_of_week]])

        print(f"[INFO] Features for IP {ip_address}: attempts={login_attempts}, "
              f"distinct_users={distinct_users}, hour={hour_of_day}, day={day_of_week}")

        if model:
            prediction = int(model.predict(features)[0])
            # predict_proba returns [[prob_class0, prob_class1]]
            confidence = float(model.predict_proba(features)[0][prediction])
        else:
            prediction = 0
            confidence = 0.0

        status = 'safe' if prediction == 0 else 'attack'

        # Response payload
        response_data = {
            'prediction': prediction,
            'status': status,
            'confidence': round(confidence, 4)
        }

        # 4. Cache the result in Redis for future requests from this IP
        limiter.cache_prediction(ip_address, response_data)

        return jsonify(response_data)

    except Exception as e:
        print(f"[ERROR] {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)