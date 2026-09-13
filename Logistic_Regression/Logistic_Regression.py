import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# X=(m,n)
# y=(m,1)
# w=(n,1)

def normalize_zscore(X):
    mu=np.mean(X,axis=0)
    std=np.std(X,axis=0)
    X_norm=(X-mu)/std
    return (X_norm,mu,std)

def compute_cost_reg(X, y, w, b,lambda_):
    m=X.shape[0]
    z = np.dot(X,w)+ b     
    f_wb = 1 / (1 + np.exp(-z))
    cost = -np.mean(y * np.log(f_wb) + (1 - y) * np.log(1 - f_wb))
    reg_cost=((lambda_)/(2*m))*np.dot(w,w)
    total_cost=cost+reg_cost
    return total_cost

def sigmoid(z):
   return 1/(1+np.exp(-z))

def compute_gradient_reg(X,y,w,b,lambda_): 
    m=X.shape[0]
    z=np.dot(X,w)+b
    f_wb=sigmoid(z)
    err=f_wb-y
    dj_dw=(1/m)*(np.dot(X.T,err))+ ((lambda_)/m)*w
    dj_db=np.mean(err)
    return (dj_dw,dj_db)

def gradient_descent(X,y,alpha=0.01,iters=1000,lambda_=1):
    w=np.zeros(X.shape[1])
    b=0
    J_history=[]
    for i in range(iters):
        dj_dw,dj_db=compute_gradient_reg(X,y,w,b,lambda_)
        w=w-(alpha)*(dj_dw)
        b=b-(alpha)*(dj_db)
        if i%100==0:
            J_history.append(compute_cost_reg(X,y,w,b,lambda_))
    return w,b,J_history

def plot_learning_curve(J_history):

#     plt.figure(figsize=(8, 6))
      plt.plot(J_history)
      plt.title("Logistic Regression (Learning Curve)")
      plt.xlabel("Iterations(per 100)")
      plt.ylabel("Cost J")
#     plt.grid(True)
      plt.show()

def predict(X,w,b):
    z=np.dot(X,w)+b
    f_wb=sigmoid(z)
    ## np.where(condition, value_if_true, value_if_false)
    y_predicted=np.where(f_wb>=0.5, 1, 0)
    return y_predicted

def accuracy(y_actual,y_predicted):

    acc=np.mean(y_actual==y_predicted)*100
    return acc

 
def save_model(filename, w, b,mu,std):
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

