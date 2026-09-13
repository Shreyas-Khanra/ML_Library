import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Assuming your knn and dist_euclid functions are in KNN.py
from KNN import * # 1. Load the data
df = pd.read_csv('train_multi_class.csv').dropna()
X = df.drop('target', axis=1).values
y = df['target'].values

# 2. Split into Train and Validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# --- NEW: NORMALIZATION STEP ---
print("Normalizing data...")
# Calculate mean and std based ONLY on training data to avoid data leakage
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)

# Avoid division by zero if a feature has zero variance
std[std == 0] = 1 

# Apply transformation: (x - mean) / std
X_train_scaled = (X_train - mean) / std
X_val_scaled = (X_val - mean) / std
# -------------------------------

print(f"Starting predictions for {len(X_val)} validation samples...")

predictions = []
for i in range(len(X_val_scaled)):
    # Predicting for the i-th validation example using the SCALED data
    pred = knn(X_train_scaled, y_train, X_val_scaled[i], k=5)
    predictions.append(pred)
    
    if (i + 1) % 1000 == 0:
        print(f"Processed {i + 1}/{len(X_val)} samples...")
        
# 4. Check accuracy
accuracy = accuracy_score(y_val, predictions)

print(f"\nFinal Results (Normalized):")
print(f"--------------------------")
print(f"Validation Accuracy: {accuracy * 100:.2f}%")