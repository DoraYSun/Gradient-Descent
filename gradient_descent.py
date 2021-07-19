# %%
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

X, y = datasets.load_boston(return_X_y=True)

# data normalization
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X = (X - X_mean) / X_std

class DataLoader:
    """creat mini batches"""
    def __init__(self, X, y, batch_size=16):
        self.batches = []
        idx = 0
        while idx < len(X):
            self.batches.append((X[idx : idx+16], y[idx : idx+16]))
            idx += batch_size
            
            
    def __getitem__(self, idx):
        return self.batches[idx]
        

train_loader = DataLoader(X, y)
train_loader[0]


class LinearRegression:
    def __init__(self, n_features):
        self.w = np.random.randn(n_features)
        self.b = np.random.randn()

    def fit(self, X, y, epochs=40):
        lr = 0.01
        losses = []
        for epoch in range(epochs):
            for X, y in train_loader:
                pred = self.predict(X)
                loss = self._get_mean_squared_error_loss(pred, y)
                losses.append(loss)
                print('Loss:', loss)
                grad_w, grab_b = self._compute_grads(X, y)
                self.w -= lr * grad_w
                self.b -= lr * grab_b
        
        plt.plot(losses)
        plt.show()
    
    def predict(self, X):
        return np.matmul(X, self.w) + self.b


    def _compute_grads(self, X, y):
        y_hat = self.predict(X)
        grab_b = 2 * np.mean(y_hat - y)

        grads_for_individual_examples = []
        for i in range(len(X)):
            grad_i = 2 * (y_hat[i] - y[i]) * X[i]
            grads_for_individual_examples.append(grad_i)
        grab_w = np.mean(grads_for_individual_examples, axis=0)
        return grab_w, grab_b
    
    def _get_mean_squared_error_loss(self, y_hat, y):
        return np.mean((y_hat - y) ** 2)

# %%
linear_model = LinearRegression(n_features=X.shape[1])
linear_model.fit(X, y)
pred = linear_model.predict(X)

# %%
