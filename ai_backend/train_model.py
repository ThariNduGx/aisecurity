import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

def train():
    print("[INFO] Loading dataset...")
    try:
        df = pd.read_csv('moodle_synthetic_login_data.csv')
    except FileNotFoundError:
        print("[ERROR] Dataset not found. Please run data_generator.py first.")
        return

    # Features and Target
    X = df[['login_attempts_last_5m', 'distinct_users_from_ip_last_5m', 'hour_of_day', 'day_of_week']]
    y = df['is_attack']

    print("[INFO] Splitting dataset into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("[INFO] Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print("[INFO] Evaluating the model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("[INFO] Exporting model to moodle_app_security_model.pkl...")
    joblib.dump(model, 'moodle_app_security_model.pkl')
    print("[SUCCESS] Model trained and saved successfully.")

if __name__ == '__main__':
    train()
