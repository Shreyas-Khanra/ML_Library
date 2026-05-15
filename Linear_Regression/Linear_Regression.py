import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def normalize_zscore(X):
    mu=np.mean(X,axis=0)
    std=np.std(X,axis=0)
    X_norm=(X-mu)/std
    return (X_norm,mu,std)

def compute_cost(X,y,w,b):
    m=X.shape[0]
    f_wb=np.dot(X,w)+b
    err=f_wb-y
    cost=np.dot(err,err)/(2*m)
    return cost

def compute_cost_reg(X,y,w,b,lambda_):
    m=X.shape[0]
    f_wb=np.dot(X,w)+b
    err=f_wb-y
    cost=np.dot(err,err)/(2*m)+(lambda_*np.dot(w,w))/(2*m)
    return cost

def compute_gradient_reg(X,y,w,b,lambda_):
    m=X.shape[0]
    f_wb=np.dot(X,w)+b
    err=f_wb-y #(m,1)
    dJ_dw=(np.dot(X.T,err))/(m) + (lambda_/m)*w
    dJ_db=np.mean(err)
    return (dJ_dw,dJ_db)

def gradient_descent(X,y,w,b,alpha=0.01,iters=5000,lambda_=10):
    J_history=[]
    for i in range(iters):
        dJ_dw,dJ_db=compute_gradient_reg(X,y,w,b,lambda_)

        w=w-(alpha)*dJ_dw
        b=b-(alpha)*dJ_db

        if i%100==0:
            J_history.append(compute_cost_reg(X,y,w,b,lambda_))

    return (w,b,J_history)

def plot_learning_curve(J_history):
    
    plt.plot(J_history)
    plt.title("Learning Curve")
    plt.xlabel("Iterations (per 100)") 
    plt.ylabel("Cost (J)")
    plt.show() 

def predict(X_new, w, b):
    
    prediction = np.dot(X_new, w) + b
    return prediction

def save_model(filename, w, b, mu, std):
    np.savez(filename, w=w, b=b,mu=mu,std=std)
    print(f"Success: Model parameters saved to '{filename}.npz'")

def load_model(filename):
        
    data = np.load(filename)
    
    # Data is stored like a dictionary
    w = data['w'] #we used keyword argument w to store w values
    b = data['b'] #we used keyword argument b to store b values
    mu=data['mu']
    std=data['std']
    print(f"Success: Model loaded from '{filename}'")
    return (w,b,mu,std)

def compute_r2_metrics(X, y, w, b):
    """
    Calculates R-squared and Adjusted R-squared for the model.
    X: Normalized feature matrix
    y: Target values
    w, b: Learned parameters
    """
    m, n = X.shape
    
    # 1. Get Predictions
    y_pred = np.dot(X, w) + b
    
    # 2. Calculate Sum of Squares Residuals (SS_res)
    ss_res = np.sum((y - y_pred)**2)
    
    # 3. Calculate Total Sum of Squares (SS_tot)
    y_mean = np.mean(y)
    ss_tot = np.sum((y - y_mean)**2)
    
    # 4. Standard R-squared
    r2 = 1 - (ss_res / ss_tot)
    
    # 5. Adjusted R-squared
    # Formula: 1 - [(1-R2)*(m-1) / (m-n-1)]
    adj_r2 = 1 - ((1 - r2) * (m - 1) / (m - n - 1))
    
    return r2, adj_r2



