import numpy as np
import matplotlib.pyplot as plt


class Linear_Regressor():

    def __init__(self, iterations=3000, learning_rate=0.01, lambda_=10):

        self.w = None
        self.b = 0
        self.mu = None
        self.std = None

        self.iterations = iterations
        self.learning_rate = learning_rate
        self.lambda_ = lambda_

        self.J_history = []

    def normalize_zscore(self, X):

        self.mu = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

        X_norm = (X - self.mu) / self.std

        return X_norm

    def compute_cost(self, X, y):

        m = X.shape[0]

        f_wb = np.dot(X, self.w) + self.b
        err = f_wb - y
        cost = np.dot(err, err) / (2 * m)
        reg_cost = (self.lambda_ * np.dot(self.w, self.w)) / (2 * m)

        return cost + reg_cost

    def compute_gradient(self, X, y):

        m = X.shape[0]
        
        f_wb = np.dot(X, self.w) + self.b
        err = f_wb - y
        dJ_dw = (np.dot(X.T, err) / m) + (self.lambda_ / m) * self.w
        dJ_db = np.mean(err)
        
        return (dJ_dw, dJ_db)

    def fit(self, X, y):

        X = self.normalize_zscore(X)

        m, n = X.shape

        self.w = np.zeros(n)
        self.b = 0

        self.J_history = []

        print("Training Model...")

        for i in range(self.iterations):

            dJ_dw, dJ_db = self.compute_gradient(X, y)
            self.w = self.w - (self.learning_rate) * dJ_dw
            self.b = self.b - (self.learning_rate) * dJ_db
            
            if i % 100 == 0:
                cost = self.compute_cost(X, y)
                self.J_history.append(cost)
                print(f"Cost after {i} iterations:{cost:.4f}")
                
        print("Training Complete!")

        return self

    def plot_learning_curve(self):

        plt.plot(self.J_history)
        plt.title("Learning Curve")
        plt.xlabel("Iterations (per 100)")
        plt.ylabel("Cost (J)")
        plt.grid(True)
        plt.show()

    def predict(self, X):

        X = (X - self.mu) / self.std
        predictions = np.dot(X, self.w) + self.b
        
        return predictions

    def r2_score(self, X, y):

        predictions = self.predict(X)

        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        return r2

    def adjusted_r2_score(self, X, y):

        m, n = X.shape

        r2 = self.r2_score(X, y)
        adj_r2 = 1 - (((1 - r2) * (m - 1)) / (m - n - 1))
        

        return adj_r2

    def save_model(self, filename):

        np.savez(
            filename,
            w=self.w,
            b=self.b,
            mu=self.mu,
            std=self.std
        )

        print(f"Model saved as {filename}.npz")

    def load_model(self, filename):

        data = np.load(filename)

        self.w = data['w']
        self.b = data['b']
        self.mu = data['mu']
        self.std = data['std']

        print("Model Loaded Successfully!")