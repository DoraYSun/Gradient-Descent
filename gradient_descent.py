# %%
from sklearn import datasets, model_selection
import numpy as np
import matplotlib.pyplot as plt


X, y = datasets.load_boston(return_X_y=True)
X_train, X_val, y_train, y_val = model_selection.train_test_split(X, y, test_size=0.2)

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

    def fit(self, X, y, epochs=400):
        lr = 0.01
        mean_training_losses = []
        mean_validation_losses = []
        for epoch in range(epochs):
            training_losses = []
            validation_losses = []
            for X, y in train_loader:
                pred = self.predict(X_train)
                pred_val = self.predict(X_val)
                training_loss = self._get_mean_squared_error_loss(pred, y_train)
                validation_loss = self._get_mean_squared_error_loss(pred_val, y_val)
                training_losses.append(training_loss)
                validation_losses.append(validation_loss)
                grad_w, grab_b = self._compute_grads(X, y)
                self.w -= lr * grad_w
                self.b -= lr * grad_b
            mean_training_losses.append(np.mean(training_losses))
            mean_validation_losses.append(np.mean(validation_losses))
            
        
        plt.plot(mean_training_losses)
        plt.plot(mean_validation_losses)
        plt.legend()
        plt.show()
    
    def predict(self, X):
        return np.matmul(X, self.w) + self.b


    def _compute_grads(self, X, y):
        y_hat = self.predict(X)
        grad_b = 2 * np.mean(y_hat - y)

        grads_for_individual_examples = []
        for i in range(len(X)):
            grad_i = 2 * (y_hat[i] - y[i]) * X[i]
            grads_for_individual_examples.append(grad_i)
        grab_w = np.mean(grads_for_individual_examples, axis=0)
        return grad_w, grad_b
    
    def _get_mean_squared_error_loss(self, y_hat, y):
        return np.mean((y_hat - y) ** 2)

# %%
linear_model = LinearRegression(n_features=X.shape[1])
linear_model.fit(X, y)
pred = linear_model.predict(X)

# %%
