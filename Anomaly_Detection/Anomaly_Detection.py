import numpy as np
import matplotlib.pyplot as plt

class Anomaly_Detection():
    def __init__(self):
        self.mu=None
        self.var=None
        self.epsilon=None
        
    def fit(self,X):
        self.mu=np.mean(X,axis=0)
        self.var=np.var(X,axis=0)
    
    def gaussian_probability(self,X):
        exponent=-(X-self.mu)**2/(2*self.var)
        gaussian=(1/np.sqrt(2*np.pi*self.var))*np.exp(exponent)
        p=np.prod(gaussian,axis=1)
        return p
    
    def predict(self,X):
        p=self.gaussian_probability(X)
        predictions=(p<self.epsilon).astype(int)
        return predictions 
    
    def F1_score(self,y_predicted,y_cv):
        
        tp = np.sum((y_cv == 1) & (y_predicted == 1))

        tn = np.sum((y_cv == 0) & (y_predicted == 0))

        fp = np.sum((y_cv == 0) & (y_predicted == 1))
        
        fn = np.sum((y_cv == 1) &(y_predicted == 0))
        
        precision = tp / (tp + fp + 1e-15)

        recall = tp / (tp + fn + 1e-15)
        
        f1 = (2 * precision * recall) / (precision + recall + 1e-15)
        
        return f1
    
    def select_threshold(self,X_cv,y_cv):
        p_cv=self.gaussian_probability(X_cv)
        best_epsilon=0
        best_f1=0
        step_size=(p_cv.max()-p_cv.min())/1000
        for epsilon in np.arange(p_cv.min(),p_cv.max(),step_size):
            predictions=(p_cv<epsilon).astype(int)
            f1_score=self.F1_score(predictions,y_cv)
            if(f1_score>best_f1):
                best_f1=f1_score
                best_epsilon=epsilon
        
        self.epsilon=best_epsilon
        
        print(f"Best Epsilon:{best_epsilon}")
        print(f"Best F1 Score:{best_f1:.4f}")
        
    
    def evaluate(self, y_actual, y_predicted):

        tp = np.sum((y_actual == 1) & (y_predicted == 1))

        tn = np.sum((y_actual == 0) & (y_predicted == 0))

        fp = np.sum((y_actual == 0) & (y_predicted == 1))
        
        fn = np.sum((y_actual == 1) &(y_predicted == 0))

        confusion_matrix = np.array([
            [tn, fp],
            [fn, tp]
        ])

        accuracy = ((tp + tn) /(tp + tn + fp + fn))

        precision = (tp /(tp + fp + 1e-15))

        recall = (tp /(tp + fn + 1e-15))

        f1_score = (2 * precision * recall) / (precision + recall + 1e-15)

        print("Confusion Matrix")
        print(confusion_matrix)

        print(f"\nAccuracy  : {accuracy*100:.2f}%")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"F1 Score  : {f1_score:.4f}")

    