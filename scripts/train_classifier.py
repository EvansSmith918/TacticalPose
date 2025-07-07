import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Paths
DATA_DIR = "data/processed"
MODEL_PATH = "models/gesture_classifier.pkl"

# Load all .npy gesture files
X = []
y = []

for filename in os.listdir(DATA_DIR):
    if filename.endswith(".npy"):
        label = filename.replace(".npy", "")
        samples = np.load(os.path.join(DATA_DIR, filename))
        X.extend(samples)
        y.extend([label] * len(samples))

# Convert to arrays
X = np.array(X)
y = np.array(y)

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(clf, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
