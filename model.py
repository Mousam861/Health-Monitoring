import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load the diabetes dataset
df = pd.read_csv('diabetes.csv')  # Make sure this file is in your working directory

# Select relevant features and target
X = df[['Glucose', 'Insulin', 'BMI', 'Age']]  # or use all 8 features
y = df['Outcome']

# Scale the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train SVC model
model = SVC(kernel='rbf', C=1, gamma='auto')  # You can experiment with kernel, C, and gamma
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model and scaler for Flask
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
