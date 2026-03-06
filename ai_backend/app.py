from flask import Flask, request, jsonify
import joblib
import numpy as np
import time
from rate_limiter import RateLimiter

app = Flask(__name__)

# Initialize the rate limiter (connects to local Redis on default port 6379)
limiter = RateLimiter(host='localhost', port=6379, db=0)

# Load the Kaggle/Colab trained UNSW-NB15 model (39 numeric features, Decision Tree)
try:
    model = joblib.load('moodle_app_security_model.pkl')
    print("[INFO] AI model (moodle_app_security_model.pkl) loaded successfully.")
except Exception as e:
    print(f"[ERROR] Could not load model: {e}")
    model = None


def get_unsw_features(ip_address, request_obj):
    """
    Build the 39-feature UNSW-NB15 vector in the exact column order the model
    was trained on (confirmed from X_train_numeric.columns in Google Colab):

      dur, spkts, dpkts, sbytes, dbytes, rate, sttl, dttl, sload, dload,
      sloss, dloss, sinpkt, dinpkt, sjit, djit, swin, stcpb, dtcpb, dwin,
      tcprtt, synack, ackdat, smean, dmean, trans_depth, response_body_len,
      ct_srv_src, ct_state_ttl, ct_dst_ltm, ct_src_dport_ltm,
      ct_dst_sport_ltm, ct_dst_src_ltm, is_ftp_login, ct_ftp_cmd,
      ct_flw_http_mthd, ct_src_ltm, ct_srv_dst, is_sm_ips_ports

    proto, state, and service are NOT included — they are categorical and
    were dropped during training (only numeric columns were kept).

    Network-layer features that cannot be measured at the HTTP application
    layer (jitter, TCP RTT, TCP sequence numbers, etc.) are set to the
    typical values for a normal Linux TCP/HTTP session.
    """
    try:
        redis = limiter.redis_client
        WINDOW = 300  # 5-minute sliding window

        # ct_src_ltm: connections from this source IP
        src_key = f"unsw:ct_src:{ip_address}"
        ct_src_ltm = redis.incr(src_key)
        if ct_src_ltm == 1:
            redis.expire(src_key, WINDOW)

        # ct_srv_src: connections from same source with same service (HTTP)
        srv_src_key = f"unsw:ct_srv_src:{ip_address}"
        ct_srv_src = redis.incr(srv_src_key)
        if ct_srv_src == 1:
            redis.expire(srv_src_key, WINDOW)

        # ct_dst_src_ltm: connections between same source/destination pair
        dst_src_key = f"unsw:ct_dst_src:{ip_address}"
        ct_dst_src_ltm = redis.incr(dst_src_key)
        if ct_dst_src_ltm == 1:
            redis.expire(dst_src_key, WINDOW)

        # ct_flw_http_mthd: HTTP GET/POST flows from this source
        http_key = f"unsw:ct_http:{ip_address}"
        ct_flw_http_mthd = redis.incr(http_key)
        if ct_flw_http_mthd == 1:
            redis.expire(http_key, WINDOW)

        # ct_state_ttl: connections with same state+TTL group from this source
        state_ttl_key = f"unsw:ct_state_ttl:{ip_address}"
        ct_state_ttl = redis.incr(state_ttl_key)
        if ct_state_ttl == 1:
            redis.expire(state_ttl_key, WINDOW)

        # ct_src_dport_ltm: same source to same dest port (80/443)
        ct_src_dport_ltm = ct_src_ltm

        # Single Moodle server = single destination
        ct_dst_ltm       = 1
        ct_srv_dst        = 1
        ct_dst_sport_ltm  = 1  # ephemeral source port not visible at app layer

    except Exception:
        ct_src_ltm = ct_srv_src = ct_dst_src_ltm = ct_flw_http_mthd = 1
        ct_state_ttl = ct_src_dport_ltm = 1
        ct_dst_ltm = ct_srv_dst = ct_dst_sport_ltm = 1

    # ── HTTP-observable payload sizes ─────────────────────────────────────────
    sbytes = int(request_obj.content_length or 0) + 200  # body + ~200 byte headers
    dbytes = 0   # response size is unknown before the model runs

    # ── Derived packet / timing estimates ─────────────────────────────────────
    dur     = 0.05                           # ~50 ms typical Moodle login round-trip
    spkts   = max(1, sbytes // 1460)         # source packets (TCP MTU = 1460 bytes)
    dpkts   = 1                              # at least 1 response packet
    rate    = (spkts + dpkts) / dur          # total packets per second
    sload   = (sbytes * 8) / dur             # source bits per second
    dload   = 0.0                            # destination load unknown pre-response
    sinpkt  = (dur * 1000) / spkts          # source inter-packet arrival time (ms)
    dinpkt  = (dur * 1000) / dpkts          # destination inter-packet time (ms)
    smean   = sbytes / spkts                 # mean source packet size (bytes)
    dmean   = 0.0                            # mean destination packet size unknown

    print(f"[INFO] UNSW features for {ip_address}: "
          f"ct_src_ltm={ct_src_ltm}, ct_srv_src={ct_srv_src}, "
          f"ct_http={ct_flw_http_mthd}, sbytes={sbytes}, rate={rate:.1f}")

    # ── 39-element feature vector — exact order from X_train_numeric.columns ──
    features = [
        dur,                #  1. dur
        spkts,              #  2. spkts
        dpkts,              #  3. dpkts
        sbytes,             #  4. sbytes
        dbytes,             #  5. dbytes
        rate,               #  6. rate            (total packets / dur)
        64,                 #  7. sttl             (Linux server default TTL)
        64,                 #  8. dttl             (client default TTL)
        sload,              #  9. sload
        dload,              # 10. dload
        0,                  # 11. sloss
        0,                  # 12. dloss
        sinpkt,             # 13. sinpkt
        dinpkt,             # 14. dinpkt
        0.0,                # 15. sjit
        0.0,                # 16. djit
        65535,              # 17. swin             (standard TCP window)
        0,                  # 18. stcpb            (TCP seq number, unknown at app layer)
        0,                  # 19. dtcpb            (TCP seq number, unknown at app layer)
        65535,              # 20. dwin
        0.0,                # 21. tcprtt
        0.0,                # 22. synack
        0.0,                # 23. ackdat
        smean,              # 24. smean
        dmean,              # 25. dmean
        1,                  # 26. trans_depth      (single HTTP request)
        0,                  # 27. response_body_len (unknown before response)
        ct_srv_src,         # 28. ct_srv_src
        ct_state_ttl,       # 29. ct_state_ttl
        ct_dst_ltm,         # 30. ct_dst_ltm
        ct_src_dport_ltm,   # 31. ct_src_dport_ltm
        ct_dst_sport_ltm,   # 32. ct_dst_sport_ltm
        ct_dst_src_ltm,     # 33. ct_dst_src_ltm
        0,                  # 34. is_ftp_login     (Moodle is not FTP)
        0,                  # 35. ct_ftp_cmd        (no FTP commands)
        ct_flw_http_mthd,   # 36. ct_flw_http_mthd
        ct_src_ltm,         # 37. ct_src_ltm
        ct_srv_dst,         # 38. ct_srv_dst
        0,                  # 39. is_sm_ips_ports  (src IP ≠ dst IP)
    ]

    return np.array([features], dtype=float)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        ip_address = data.get('ip', request.remote_addr)

        # 1. Rate Limiting Check
        if limiter.is_rate_limited(ip_address):
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

        # 3. Build 39-feature UNSW-NB15 vector and run the trained model
        features = get_unsw_features(ip_address, request)

        if model:
            prediction = int(model.predict(features)[0])
            confidence = float(model.predict_proba(features)[0][prediction])
        else:
            prediction = 0
            confidence = 0.0

        status = 'safe' if prediction == 0 else 'attack'

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
