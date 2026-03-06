from flask import Flask, request, jsonify
import joblib
import numpy as np
import time
from rate_limiter import RateLimiter

app = Flask(__name__)

# Initialize the rate limiter (connects to local Redis on default port 6379)
limiter = RateLimiter(host='localhost', port=6379, db=0)

# Load the UNSW-NB15 trained AI model (39 purely numeric features, Decision Tree)
# Place moodle_ai_security_model.pkl (downloaded from Colab) in this folder.
try:
    model = joblib.load('moodle_ai_security_model.pkl')
    print("[INFO] UNSW-NB15 AI model (moodle_ai_security_model.pkl) loaded successfully.")
except Exception as e:
    print(f"[ERROR] Could not load UNSW-NB15 model: {e}")
    print("[ERROR] Upload moodle_ai_security_model.pkl from your Google Colab download.")
    model = None


# ──────────────────────────────────────────────────────────────────────────────
# IMPORTANT: Feature order must match X_train_numeric.columns from your Colab.
# Run this one-liner in your Colab notebook to confirm the exact order:
#   print(list(X_train_numeric.columns))
# Then update the feature list in get_unsw_features() below to match.
# ──────────────────────────────────────────────────────────────────────────────


def get_unsw_features(ip_address, request_obj):
    """
    Build the 39 purely numeric UNSW-NB15 features from what is observable
    at the HTTP application layer.

    proto, state, and service are NOT included — they were dropped during
    training (X_train_numeric contained only numeric columns).

    Assumed 39-feature column order (after dropping srcip, dstip, proto,
    state, service, stcpb, dtcpb, Stime, attack_cat, Label from the 49-column
    UNSW-NB15 CSV):

      sport, dsport, dur, sbytes, dbytes, sttl, dttl, sloss, dloss,
      Sload, Dload, Spkts, Dpkts, swin, dwin, smeansz, dmeansz,
      trans_depth, res_bdy_len, Sjit, Djit, Ltime, Sintpkt, Dintpkt,
      tcprtt, synack, ackdat, is_sm_ips_ports, ct_state_ttl,
      ct_flw_http_mthd, is_ftp_login, ct_ftp_cmd, ct_srv_src, ct_srv_dst,
      ct_dst_ltm, ct_src_ltm, ct_src_dport_ltm, ct_dst_sport_ltm,
      ct_dst_src_ltm
    """
    try:
        redis = limiter.redis_client
        WINDOW = 300  # 5-minute sliding window

        # ct_src_ltm – connections from this source IP in the window
        src_key = f"unsw:ct_src:{ip_address}"
        ct_src_ltm = redis.incr(src_key)
        if ct_src_ltm == 1:
            redis.expire(src_key, WINDOW)

        # ct_srv_src – connections from same source with HTTP service
        srv_src_key = f"unsw:ct_srv_src:{ip_address}"
        ct_srv_src = redis.incr(srv_src_key)
        if ct_srv_src == 1:
            redis.expire(srv_src_key, WINDOW)

        # ct_dst_src_ltm – connections between same source/destination pair
        dst_src_key = f"unsw:ct_dst_src:{ip_address}"
        ct_dst_src_ltm = redis.incr(dst_src_key)
        if ct_dst_src_ltm == 1:
            redis.expire(dst_src_key, WINDOW)

        # ct_flw_http_mthd – HTTP GET/POST flows from this source
        http_key = f"unsw:ct_http:{ip_address}"
        ct_flw_http_mthd = redis.incr(http_key)
        if ct_flw_http_mthd == 1:
            redis.expire(http_key, WINDOW)

        # ct_state_ttl – connections with same state+TTL from this source
        state_ttl_key = f"unsw:ct_state_ttl:{ip_address}"
        ct_state_ttl = redis.incr(state_ttl_key)
        if ct_state_ttl == 1:
            redis.expire(state_ttl_key, WINDOW)

        # ct_src_dport_ltm: same source to same dest port (80/443)
        ct_src_dport_ltm = ct_src_ltm

        # Single Moodle server = single destination
        ct_srv_dst       = 1
        ct_dst_ltm       = 1
        ct_dst_sport_ltm = 1  # source port not observable at application layer

    except Exception:
        ct_src_ltm = ct_srv_src = ct_dst_src_ltm = ct_flw_http_mthd = 1
        ct_state_ttl = ct_src_dport_ltm = 1
        ct_srv_dst = ct_dst_ltm = ct_dst_sport_ltm = 1

    # ── Port numbers ──────────────────────────────────────────────────────────
    sport  = 0    # ephemeral source port not visible at HTTP app layer
    dsport = 80   # Moodle destination port (HTTP); change to 443 for HTTPS

    # ── HTTP-observable payload sizes ─────────────────────────────────────────
    sbytes = int(request_obj.content_length or 0) + 200  # body + ~200 byte headers
    dbytes = 0   # response size unknown before prediction

    # ── Derived packet / timing estimates ─────────────────────────────────────
    dur     = 0.05                           # ~50 ms typical Moodle login round-trip
    spkts   = max(1, sbytes // 1460)         # packets (MTU = 1460 bytes)
    dpkts   = 1                              # at least 1 response packet
    sload   = (sbytes * 8) / dur             # source bits per second
    dload   = 0.0                            # response load unknown pre-response
    smeansz = sbytes / spkts                 # mean source packet size
    dmeansz = 0.0                            # mean destination packet size unknown
    sintpkt = (dur * 1000) / spkts          # source inter-packet arrival time (ms)
    dintpkt = dur * 1000                     # destination inter-packet time (ms)
    ltime   = time.time()                    # Unix timestamp of last packet

    print(f"[INFO] UNSW features for {ip_address}: "
          f"ct_src_ltm={ct_src_ltm}, ct_srv_src={ct_srv_src}, "
          f"ct_http={ct_flw_http_mthd}, sbytes={sbytes}")

    # ── 39-element feature vector (no categorical columns) ────────────────────
    # proto, state, service are intentionally absent — they were dropped during
    # training. Only the numeric columns from X_train_numeric are included here.
    features = [
        sport,              #  1. sport          (numeric port number)
        dsport,             #  2. dsport          (numeric port number)
        dur,                #  3. dur
        sbytes,             #  4. sbytes
        dbytes,             #  5. dbytes
        64,                 #  6. sttl            (Linux default TTL)
        64,                 #  7. dttl
        0,                  #  8. sloss
        0,                  #  9. dloss
        sload,              # 10. Sload
        dload,              # 11. Dload
        spkts,              # 12. Spkts
        dpkts,              # 13. Dpkts
        65535,              # 14. swin            (standard TCP window size)
        65535,              # 15. dwin
        smeansz,            # 16. smeansz
        dmeansz,            # 17. dmeansz
        1,                  # 18. trans_depth     (single HTTP request)
        0,                  # 19. res_bdy_len     (unknown before response)
        0.0,                # 20. Sjit
        0.0,                # 21. Djit
        ltime,              # 22. Ltime           (current Unix timestamp)
        sintpkt,            # 23. Sintpkt
        dintpkt,            # 24. Dintpkt
        0.0,                # 25. tcprtt
        0.0,                # 26. synack
        0.0,                # 27. ackdat
        0,                  # 28. is_sm_ips_ports (src IP ≠ dst IP)
        ct_state_ttl,       # 29. ct_state_ttl    (live Redis counter)
        ct_flw_http_mthd,   # 30. ct_flw_http_mthd
        0,                  # 31. is_ftp_login    (Moodle is not FTP)
        0,                  # 32. ct_ftp_cmd      (no FTP commands)
        ct_srv_src,         # 33. ct_srv_src
        ct_srv_dst,         # 34. ct_srv_dst
        ct_dst_ltm,         # 35. ct_dst_ltm
        ct_src_ltm,         # 36. ct_src_ltm
        ct_src_dport_ltm,   # 37. ct_src_dport_ltm
        ct_dst_sport_ltm,   # 38. ct_dst_sport_ltm
        ct_dst_src_ltm,     # 39. ct_dst_src_ltm
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
