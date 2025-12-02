import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv('Task 3 and 4_Loan_Data.csv')

X = data.drop(['customer_id', 'default'], axis=1)
y = data['default']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

def predict_default_probability(loan_features):
    features_array = np.array(loan_features).reshape(1, -1)
    features_scaled = scaler.transform(features_array)
    default_prob = rf_model.predict_proba(features_scaled)[0][1]
    return default_prob

def calculate_expected_loss(loan_features, loan_amount, recovery_rate=0.1):
    pd_value = predict_default_probability(loan_features)
    expected_loss = pd_value * loan_amount * (1 - recovery_rate)
    return expected_loss



print("=== Loan Default Prediction ===")

sample_features = [50000, 2.5, 650, 1, 0, 28]  # Example features
loan_amount = 10000


prob_default = predict_default_probability(sample_features)
expected_loss = calculate_expected_loss(sample_features, loan_amount)

print(f"Probability of Default: {prob_default:.2%}")
print(f"Expected Loss: ${expected_loss:.2f}")
print(f"Loan Amount: ${loan_amount:,.2f}")

y_pred = rf_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2%}")