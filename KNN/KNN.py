import numpy as np 
import matplotlib.pyplot as plt

# X=(m,n) m=examples and n dimensions
# y=(m,1) y=labels 
# eg=(1,n) eg=the example which we want to label

def dist_euclid(X,eg):
    #   dist=np.sqrt(np.sum(temp**2,axis=1)) we can skip np.sort because a>b means sqrt(a)>sqrt(b)
    dist_temp=np.sum((X-eg)**2,axis=1)
    return dist_temp

def knn(X,y,eg,k=3):
    dist_proxy=dist_euclid(X,eg)

    # Get the indices of lowest distances
    #np.argsort() sorts default in ascending order
    k_indices=np.argsort(dist_proxy)[:k]

    #labels of KNN
    k_labels=y[k_indices]
    
    vals, counts = np.unique(k_labels, return_counts=True)
    return vals[np.argmax(counts)]
    
