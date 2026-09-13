import numpy as np
import random

def compute_cost(X,mu_assigned):
    cost=np.mean(np.sum((X-mu_assigned)**2,axis=1))
    return cost

def initialize(X,k):
    indices=random.sample(range(0,X.shape[0]),k)
    mu=X[indices]
    return mu

def find_closest_centroids(X,mu):
    K=mu.shape[0]  
    #mu=(k,n)
    #X=(m,n)
    #(m,1,n) broadcasted to (m,k,n) - (k,n)
    idx=np.zeros(X.shape[0],dtype=int)
    distances=np.sum((X[:,np.newaxis]-mu)**2,axis=2)
    idx=np.argmin(distances,axis=1)
    return idx

def update_centroids(X,idx,mu):
    for i in range(mu.shape[0]):
        points_in_cluster=X[idx==i]
        if points_in_cluster.shape[0]>0:
            mu[i]=np.mean(points_in_cluster,axis=0)
    return mu

def run_kmeans(X,k,iters):
    mu=initialize(X,k)
    for i in range(iters):
        idx=find_closest_centroids(X,mu)
        mu=update_centroids(X,idx,mu)
    
    return mu,idx

def select_best_model(X,k,iters):
    cost_values={}
    best_mu=[]
    for i in range(5): 
        mu,idx=run_kmeans(X,k,iters)

        assigned_mu=mu[idx]

        cost_values[compute_cost(X,assigned_mu)]=(mu,idx)
    best_mu,best_idx=cost_values[min(cost_values)]
    return best_mu,best_idx,min(cost_values)

