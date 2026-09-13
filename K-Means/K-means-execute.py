import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from K_Means import *
from sklearn.decomposition import PCA

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

k=int(input("Enter optimal value of k from Elbow Plot:"))

best_mu, best_idx,_ = select_best_model(X, k, iters=20)

#PCA Transformation
# We reduce 6 features down to 2 components for plotting
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
# Important: Centroids must be transformed by the same PCA
centroids_pca = pca.transform(best_mu)

# 4. Generate Plot
plt.figure(figsize=(10, 7))
# Plot data points colored by their cluster ID
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=best_idx, cmap='viridis', s=5, alpha=0.4)
# Plot transformed centroids
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
            c='red', marker='X', s=150, label='Centroids', edgecolors='black')

plt.title(f'PCA Visualization of K-Means Clusters (K={k})')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('pca_cluster_plot.png')

