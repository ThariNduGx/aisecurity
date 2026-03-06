import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_synthetic_data(num_records=5000):
    """
    Generates a synthetic dataset predicting application-layer Web attacks (like brute force)
    on a Moodle LMS based on typical web application metrics.
    """
    data = []
    
    # Define common ranges
    normal_hours = list(range(8, 20)) # 8 AM to 8 PM
    odd_hours = list(range(0, 8)) + list(range(20, 24))
    
    for _ in range(num_records):
        # 80% Normal Traffic, 20% Anomalous/Attack Traffic
        is_attack = 1 if random.random() < 0.2 else 0
        
        if is_attack == 0:
            # --- NORMAL LOGIN BEHAVIOR ---
            # Few attempts in the last 5 minutes (user might have forgotten password once or twice)
            login_attempts_last_5m = random.choices([1, 2, 3], weights=[0.8, 0.15, 0.05])[0]
            
            # Normal users only try 1 account from their IP
            distinct_users_from_ip_last_5m = 1
            
            # Mostly during normal hours
            hour_of_day = random.choice(normal_hours) if random.random() < 0.9 else random.choice(odd_hours)
            
            # Mostly weekdays
            day_of_week = random.choices([0, 1, 2, 3, 4, 5, 6], weights=[0.18, 0.18, 0.18, 0.18, 0.18, 0.05, 0.05])[0]
            
        else:
            # --- ATTACK BEHAVIOR ---
            attack_type = random.choice(['brute_force', 'credential_stuffing', 'odd_time_spike'])
            
            if attack_type == 'brute_force':
                # High number of attempts for the same user from the same IP
                login_attempts_last_5m = random.randint(15, 100)
                distinct_users_from_ip_last_5m = random.choices([1, 2], weights=[0.9, 0.1])[0]
                hour_of_day = random.choice(range(0, 24))
                day_of_week = random.choice(range(0, 7))
                
            elif attack_type == 'credential_stuffing':
                # Normal-ish attempt rates but spread across MANY different user IDs
                login_attempts_last_5m = random.randint(10, 50)
                distinct_users_from_ip_last_5m = random.randint(10, 50)
                hour_of_day = random.choice(odd_hours)
                day_of_week = random.choice(range(0, 7))
                
            elif attack_type == 'odd_time_spike':
                # Sudden burst of activity at 3 AM
                login_attempts_last_5m = random.randint(5, 20)
                distinct_users_from_ip_last_5m = random.randint(1, 5)
                hour_of_day = random.choice(odd_hours)
                day_of_week = random.choice([5, 6]) # Weekend night

        data.append({
            'login_attempts_last_5m': login_attempts_last_5m,
            'distinct_users_from_ip_last_5m': distinct_users_from_ip_last_5m,
            'hour_of_day': hour_of_day,
            'day_of_week': day_of_week,
            'is_attack': is_attack
        })
        
    df = pd.DataFrame(data)
    
    # Save the dataset
    df.to_csv('moodle_synthetic_login_data.csv', index=False)
    print(f"[INFO] Generated synthetic dataset with {len(df)} records. Saved to 'moodle_synthetic_login_data.csv'.")
    print(df['is_attack'].value_counts())

if __name__ == "__main__":
    generate_synthetic_data(10000)
