import numpy as np
import matplotlib.pyplot as plt

# X=(m,n)
# y=(m,1)
# w=(n,1)

class Logistic_Regression():
    def __init__(self,iterations=3000,learning_rate=0.1,lambda_=10):
        self.mu=None
        self.std=None
        self.w=None
        self.b=0
        self.iterations=iterations
        self.learning_rate=learning_rate
        self.lambda_=lambda_
        self.J_history=[]
        
    def normalize_zscore(self,X):
        self.mu=np.mean(X,axis=0)
        self.std=np.std(X,axis=0)
        X_norm=(X-self.mu)/self.std
        return X_norm
    
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    
    def compute_cost(self,X,y):
        m=X.shape[0]
        z=np.dot(X,self.w)+self.b
        f_wb=self.sigmoid(z)
        cost=-np.mean(y*np.log(f_wb)+(1-y)*np.log(1-f_wb))
        reg_cost=((self.lambda_)/(2*m))*np.dot(self.w,self.w)
        return cost+reg_cost
    
    def compute_gradient(self,X,y):
        m=X.shape[0]
        z=np.dot(X,self.w)+self.b
        f_wb=self.sigmoid(z)
        err=f_wb-y
        dj_dw=(1/m)*(np.dot(X.T,err))+((self.lambda_)/m)*self.w
        dj_db=np.mean(err)
        return (dj_dw,dj_db)
    
    def fit(self,X,y):
        X=self.normalize_zscore(X)
        self.w=np.zeros(X.shape[1])
        self.b=0
        self.J_history=[]
        
        print("Training Model...")
        for i in range(self.iterations):
            dj_dw,dj_db=self.compute_gradient(X,y) 
            
            self.w=self.w-(self.learning_rate)*(dj_dw)
            self.b=self.b-(self.learning_rate)*(dj_db)
            
            if i%100==0:
                cost=self.compute_cost(X,y)
                print(f"Cost after {i} iterations:{cost:.4f}") 
                self.J_history.append(cost)
    
    def plot_learning_curve(self):
        
        plt.plot(self.J_history)
        plt.title("Learning Curve")
        plt.xlabel("Iterations(per 100)")
        plt.ylabel("Cost(J)")
        plt.grid(True)
        plt.show()
        
    def predict(self,X,threshold=0.5):
        X=(X-self.mu)/self.std
        z=np.dot(X,self.w)+self.b
        f_wb=self.sigmoid(z)
        y_predicted=np.where(f_wb>=threshold,1,0)
        return y_predicted
    
    def evaluate(self, y_actual, y_predicted):

        # True Positives
        tp = np.sum((y_actual == 1) & (y_predicted == 1))

        # True Negatives
        tn = np.sum((y_actual == 0) & (y_predicted == 0))

        # False Positives
        fp = np.sum((y_actual == 0) & (y_predicted == 1))

        # False Negatives
        fn = np.sum((y_actual == 1) & (y_predicted == 0))

        # Confusion Matrix
        cm = np.array([
            [tn, fp],
            [fn, tp]
        ])

        # Accuracy
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        # Precision
        precision = tp / (tp + fp + 1e-15)
        
        # Recall
        recall = tp / (tp + fn + 1e-15)

        # F1 Score
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-15)

        # Print Results
        print("Confusion Matrix")
        print(cm)

        print(f"\nAccuracy  : {accuracy*100:.2f}%")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"F1 Score  : {f1_score:.4f}")

        
        
    def save_model(self,filename):
        np.savez(filename, w=self.w, b=self.b,mu=self.mu,std=self.std)
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
