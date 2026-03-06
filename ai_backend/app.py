from flask import Flask, request, jsonify
import joblib
import numpy as np
import datetime
from rate_limiter import RateLimiter

app = Flask(__name__)

# Initialize the rate limiter (connects to local Redis on default port 6379)
limiter = RateLimiter(host='localhost', port=6379, db=0)

# Load the UNSW-NB15 trained AI model (39 features, Decision Tree from Google Colab)
# Place moodle_ai_security_model.pkl (downloaded from Colab) in this folder.
try:
    model = joblib.load('moodle_ai_security_model.pkl')
    print("[INFO] UNSW-NB15 AI model (moodle_ai_security_model.pkl) loaded successfully.")
except Exception as e:
    print(f"[ERROR] Could not load UNSW-NB15 model: {e}")
    print("[ERROR] Upload moodle_ai_security_model.pkl from your Google Colab download.")
    model = None

# ─── UNSW-NB15 Label Encoding Constants ───────────────────────────────────────
# These values match sklearn LabelEncoder applied with alphabetical ordering
# to the unique values present in the UNSW-NB15 Kaggle dataset.
#
# proto (alphabetically: arp=0 … tcp=27, udp=28 …)
PROTO_TCP = 27
#
# service (alphabetically: '-'=0, dhcp=1, dns=2, ftp=3, ftp-data=4,
#          http=5, irc=6, pop3=7, radius=8, smtp=9, snmp=10, ssh=11, ssl=12)
SERVICE_HTTP = 5
#
# state (alphabetically: ACC=0, CLO=1, CON=2, ECO=3, ECOA=4,
#        FIN=5, INT=6, MAS=7, PAR=8, REQ=9, RST=10, …)
STATE_FIN = 5
#
# ct_state_ttl: UNSW-NB15 assigns 2 for FIN state with TTL in the 32-64 range
# (Linux web server default TTL=64 falls in this band)
CT_STATE_TTL_HTTP = 2


def get_unsw_features(ip_address, request_obj):
    """
    Build the 39-feature UNSW-NB15 vector from what is observable at the
    HTTP application layer.

    Network-layer features that cannot be measured at the app layer
    (jitter, TCP RTT, window sizes, packet loss, etc.) are set to the
    typical values for a normal Linux TCP/HTTP session so the model
    baseline is correct and unbiased toward 'attack'.

    Feature order matches UNSW-NB15 CSV column order after dropping:
      srcip, sport, dstip, dsport, stcpb, dtcpb, Stime, Ltime,
      attack_cat, Label  (10 columns removed → 39 remain)

    The 39 features in order:
      proto, state, dur, sbytes, dbytes, sttl, dttl, sloss, dloss,
      service, Sload, Dload, Spkts, Dpkts, swin, dwin, smeansz, dmeansz,
      trans_depth, res_bdy_len, Sjit, Djit, Sintpkt, Dintpkt, tcprtt,
      synack, ackdat, is_sm_ips_ports, ct_state_ttl, ct_flw_http_mthd,
      is_ftp_login, ct_ftp_cmd, ct_srv_src, ct_srv_dst, ct_dst_ltm,
      ct_src_ltm, ct_src_dport_ltm, ct_dst_sport_ltm, ct_dst_src_ltm
    """
    try:
        redis = limiter.redis_client
        WINDOW = 300  # 5-minute sliding window (approximates UNSW-NB15's 100-conn window)

        # ct_src_ltm – connections from this source IP
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

        # ct_src_dport_ltm: same source → same dest port (80/443), mirrors ct_src_ltm
        ct_src_dport_ltm = ct_src_ltm

        # Single Moodle server = single destination → these are stable at 1
        ct_srv_dst       = 1
        ct_dst_ltm       = 1
        ct_dst_sport_ltm = 1  # can't observe source port at application layer

    except Exception:
        ct_src_ltm = ct_srv_src = ct_dst_src_ltm = ct_flw_http_mthd = 1
        ct_src_dport_ltm = ct_srv_dst = ct_dst_ltm = ct_dst_sport_ltm = 1

    # ── HTTP-observable payload sizes ─────────────────────────────────────────
    sbytes = int(request_obj.content_length or 0) + 200  # body + ~200 byte headers
    dbytes = 0  # response size unknown before the model runs; use 0

    # ── Derived packet/timing estimates ───────────────────────────────────────
    dur     = 0.05              # ~50 ms typical Moodle login round-trip (seconds)
    spkts   = max(1, sbytes // 1460)            # approx packets (MTU = 1460 bytes)
    dpkts   = 1                                 # at least 1 response packet
    sload   = (sbytes * 8) / dur                # source bits per second
    dload   = 0.0                               # response load unknown
    smeansz = sbytes / spkts                    # mean src packet size
    dmeansz = 0.0                               # mean dst packet size unknown
    sintpkt = (dur * 1000) / spkts             # src inter-packet time (ms)
    dintpkt = dur * 1000                        # dst inter-packet time (ms)

    print(f"[INFO] UNSW features for {ip_address}: ct_src={ct_src_ltm}, "
          f"ct_srv_src={ct_srv_src}, ct_http={ct_flw_http_mthd}, sbytes={sbytes}")

    # ── 39-element feature vector (must match training column order exactly) ──
    features = [
        PROTO_TCP,          #  1. proto
        STATE_FIN,          #  2. state
        dur,                #  3. dur
        sbytes,             #  4. sbytes
        dbytes,             #  5. dbytes
        64,                 #  6. sttl  (Linux default TTL)
        64,                 #  7. dttl
        0,                  #  8. sloss
        0,                  #  9. dloss
        SERVICE_HTTP,       # 10. service
        sload,              # 11. Sload
        dload,              # 12. Dload
        spkts,              # 13. Spkts
        dpkts,              # 14. Dpkts
        65535,              # 15. swin  (standard TCP window)
        65535,              # 16. dwin
        smeansz,            # 17. smeansz
        dmeansz,            # 18. dmeansz
        1,                  # 19. trans_depth (single HTTP request)
        0,                  # 20. res_bdy_len (unknown before response)
        0.0,                # 21. Sjit
        0.0,                # 22. Djit
        sintpkt,            # 23. Sintpkt
        dintpkt,            # 24. Dintpkt
        0.0,                # 25. tcprtt
        0.0,                # 26. synack
        0.0,                # 27. ackdat
        0,                  # 28. is_sm_ips_ports  (src IP ≠ dst IP)
        CT_STATE_TTL_HTTP,  # 29. ct_state_ttl
        ct_flw_http_mthd,   # 30. ct_flw_http_mthd
        0,                  # 31. is_ftp_login  (Moodle is not FTP)
        0,                  # 32. ct_ftp_cmd    (no FTP commands)
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
