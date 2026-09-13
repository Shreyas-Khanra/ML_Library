import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from K_Means import *

df = pd.read_csv('unsupervised_data.csv')
X = df.drop(columns=['ID']).values

k_values = range(1, 11)
costs = []

print("Searching for optimal k...")
for k in k_values:
    _, _, cost = select_best_model(X, k, iters=10)
    costs.append(cost)
    print(f"k={k} | Cost: {cost:.2f}")

# Plotting  
plt.plot(k_values, costs,'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Cost (Distortion)')
plt.title('Elbow Method for Optimal k')
plt.xticks(k_values)
plt.grid(True)
plt.savefig('elbow_plot.png')

