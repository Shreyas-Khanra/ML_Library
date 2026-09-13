import numpy as np
import pandas as pd
from Decision_Tree import *

# --- 2. DATA LOADING & SPLITTING ---
# Load and clean (Important: dropping NAs as identified in your CSV)
df = pd.read_csv('train_multi_class.csv').dropna()
X = df.drop('target', axis=1).values
y = df['target'].values.astype(int)

# Manual 80/20 Shuffle Split
indices = np.arange(len(X))
np.random.shuffle(indices)
split_limit = int(0.8 * len(X))

train_idx, val_idx = indices[:split_limit], indices[split_limit:]
X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]

# --- 3. TRAINING & EVALUATION ---
# Using max_depth=10 to keep it fast while maintaining complexity
clf = DecisionTree(max_depth=12,min_samples_split=20, n_bins=100)
clf.fit(X_train, y_train)

# Calculate accuracy on the 20% validation set
y_pred = clf.predict(X_val)
accuracy = np.mean(y_val == y_pred)

print(f"Training Complete!")
print(f"Validation Accuracy: {accuracy * 100:.2f}%")