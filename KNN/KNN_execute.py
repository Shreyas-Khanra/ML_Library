import pandas as pd
import numpy as np
from KNN import *

# Loading the data
df_train= pd.read_csv('train_multi_class.csv').dropna()
X_train = df_train.drop('target', axis=1).values
y_train = df_train['target'].values

df_test=pd.read_csv('test_multi_class.csv')
X_test = df_test.values
print(f"Starting predictions for {len(X_test)} samples...")

# We use a loop to process each 'eg' in the test set
predictions = []
for i in range(len(X_test)):
    # Predicting for the i-th test example
    pred = knn(X_train, y_train, X_test[i], k=5)
    predictions.append(pred)
    
    # Optional: Print progress every 1000 samples
    if (i + 1) % 1000 == 0:
        print(f"Processed {i + 1}/{len(X_test)} samples...")
        
output=pd.DataFrame({"Predictions":predictions})
output.to_csv("Predictions.csv",index=False)

print("Final results saved to Predictions.csv")
