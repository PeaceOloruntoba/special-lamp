import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify

app = Flask(__name__)

# Preprocess data
def preprocess_data(data):
    scaler = StandardScaler()
    # Convert IP to numeric (simplified)
    data['ip_numeric'] = data['ip'].apply(lambda x: int(''.join([f'{int(o):03d}' for o in x.split('.')])))
    # Hash device_id and location for numerical representation
    data['device_id_hash'] = data['device_id'].apply(lambda x: hash(x) % 10**8)
    data['location_hash'] = data['location'].apply(lambda x: hash(x) % 10**8)
    # Normalize features
    features = data[['timestamp', 'ip_numeric', 'device_id_hash', 'location_hash']]
    return scaler.fit_transform(features), scaler

# Train anomaly detection model
def train_anomaly_model(data):
    features, _ = preprocess_data(data)
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(features)
    return model

# Train classification model
def train_classifier(data):
    features, _ = preprocess_data(data)
    labels = data['is_fraud']
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(features, labels)
    return classifier

# Detect fraud
def detect_fraud(anomaly_model, classifier, scaler, vote_data):
    vote_data = vote_data.copy()
    # Preprocess single vote
    vote_data['ip_numeric'] = vote_data['ip'].apply(lambda x: int(''.join([f'{int(o):03d}' for o in x.split('.')])))
    vote_data['device_id_hash'] = vote_data['device_id'].apply(lambda x: hash(x) % 10**8)
    vote_data['location_hash'] = vote_data['location'].apply(lambda x: hash(x) % 10**8)
    features = vote_data[['timestamp', 'ip_numeric', 'device_id_hash', 'location_hash']]
    scaled_features = scaler.transform(features)
    
    # Check for anomaly
    anomaly = anomaly_model.predict(scaled_features)
    if anomaly[0] == -1:
        # Confirm with classifier
        fraud_prob = classifier.predict_proba(scaled_features)[0][1]
        return fraud_prob > 0.7, fraud_prob
    return False, 0.0

# Load and train models
data = pd.read_csv('voting_data.csv')  # Simulated dataset
anomaly_model = train_anomaly_model(data)
classifier = train_classifier(data)
_, scaler = preprocess_data(data)

# Flask API for fraud detection
@app.route('/api/detect_fraud', methods=['POST'])
def api_detect_fraud():
    vote_data = request.json
    df = pd.DataFrame([vote_data])
    is_fraud, fraud_prob = detect_fraud(anomaly_model, classifier, scaler, df)
    return jsonify({'is_fraud': is_fraud, 'fraud_probability': fraud_prob})

if __name__ == '__main__':
    app.run(port=5000, debug=True)

# Example simulated dataset generation
def generate_simulated_data(n_samples=1000):
    np.random.seed(42)
    data = {
        'voter_id': [f'voter_{i}' for i in range(n_samples)],
        'timestamp': np.random.randint(1634567890, 1634654290, n_samples),
        'ip': [f'192.168.1.{np.random.randint(1, 255)}' for _ in range(n_samples)],
        'device_id': [f'device_{np.random.randint(1000, 9999)}' for _ in range(n_samples)],
        'location': np.random.choice(['Lagos', 'Abuja', 'Kano', 'Ibadan'], n_samples),
        'is_fraud': np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
    }
    pd.DataFrame(data).to_csv('voting_data.csv', index=False)

# Uncomment to generate data
# generate_simulated_data()
