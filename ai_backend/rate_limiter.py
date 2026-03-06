import redis
import json

class RateLimiter:
    def __init__(self, host='localhost', port=6379, db=0):
        # Connect to the local Redis server
        self.redis_client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        self.MAX_REQUESTS = 10  # Max requests allowed before blocking
        self.TIME_WINDOW = 60   # The time window to track requests (seconds)
        self.BLOCK_TIME = 300   # How long to block the IP if threshold exceeded (seconds)
        self.CACHE_TIME = 120   # How long to cache AI predictions (seconds)

    def is_rate_limited(self, ip_address):
        """
        Checks if an IP address should be rate limited.
        Returns True if limited (blocked), False otherwise.
        """
        try:
            block_key = f"blocked:{ip_address}"
            req_key = f"requests:{ip_address}"

            # 1. Check if the IP is currently serving a block penalty
            if self.redis_client.exists(block_key):
                return True

            # 2. Increment request count for this IP
            # If key doesn't exist, it creates it and sets the value to 1
            current_requests = self.redis_client.incr(req_key)

            # 3. Set expiration for the time window on the first request
            if current_requests == 1:
                self.redis_client.expire(req_key, self.TIME_WINDOW)

            # 4. Check if the limit is exceeded (High velocity/Volumetric attack)
            if current_requests > self.MAX_REQUESTS:
                # Issue a block penalty
                self.redis_client.setex(block_key, self.BLOCK_TIME, "blocked")
                return True

            return False
            
        except redis.ConnectionError:
            # If Redis crashes or isn't running, fail open to avoid taking Moodle offline
            print("[WARNING] Redis connection failed. Bypassing rate limiter.")
            return False

    def get_cached_prediction(self, ip_address):
        """
        Retrieves a previously cached AI prediction for this IP to reduce latency.
        """
        try:
            cache_key = f"prediction_cache:{ip_address}"
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
        except redis.ConnectionError:
            pass
        return None

    def cache_prediction(self, ip_address, prediction_data):
        """
        Stores the AI prediction in Redis.
        """
        try:
            cache_key = f"prediction_cache:{ip_address}"
            self.redis_client.setex(cache_key, self.CACHE_TIME, json.dumps(prediction_data))
        except redis.ConnectionError:
            pass
