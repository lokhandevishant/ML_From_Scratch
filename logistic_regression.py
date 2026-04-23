import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    # np.clip is highly recommended to prevent overflow errors with np.exp()
    # It stops numbers from getting too astronomically large or small.
    z = np.clip(z, -250, 250) 
    return 1 / (1 + np.exp(-z))

class LogisticRegression():
    def __init__(self, learning_rate=0.01, iterations = 1000):
        self.weight = None
        self.bias = None 
        self.lr = learning_rate
        self.iterations = iterations

    def fit(self, X, y):
        # n_samples = rows, n_features = columns
        n_samples, n_features = X.shape 
        self.weight = np.zeros(n_features)
        self.bias = 0 

        for _ in range(self.iterations):
            #Linear Regression Formula: y = wx + b
            y_predicted = sigmoid(np.dot(X, self.weight)+ self.bias)

            #Calculate Gradients (Derivative) 
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)

            #Update Parameters
            self.weight = self.weight - (self.lr * dw) 
            self.bias = self.bias - (self.lr * db) 

    def predict(self, X):
        return sigmoid(np.dot(X, self.weight) + self.bias)
    
    
    



'''Testing'''
# --- 1. Generate Synthetic Classification Data ---
np.random.seed(42)

# Class 0: 100 points centered around (2, 2)
X_0 = np.random.randn(100, 2) + np.array([2, 2])
y_0 = np.zeros(100)

# Class 1: 100 points centered around (6, 6)
X_1 = np.random.randn(100, 2) + np.array([6, 6])
y_1 = np.ones(100)

# Combine and shuffle
X = np.vstack((X_0, X_1))
y = np.hstack((y_0, y_1))

shuffle_idx = np.random.permutation(len(X))
X = X[shuffle_idx]
y = y[shuffle_idx]

# --- 2. Train the Model ---
model = LogisticRegression(learning_rate=0.05, iterations=1000)
model.fit(X, y)
y_pred_probs = model.predict(X)
y_pred_labels = (y_pred_probs >= 0.5).astype(int)

# --- 3. Evaluate Accuracy ---
acc = np.mean(y_pred_labels == y)
print(f"Model Accuracy: {acc * 100:.2f}%")

# --- 4. Plot the Results ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(X[y==0, 0], X[y==0, 1], color='blue', label='Class 0')
ax1.scatter(X[y==1, 0], X[y==1, 1], color='red', label='Class 1')
ax1.set_title("True Labels")
ax1.legend()

ax2.scatter(X[y_pred_labels==0, 0], X[y_pred_labels==0, 1], color='blue', label='Pred Class 0')
ax2.scatter(X[y_pred_labels==1, 0], X[y_pred_labels==1, 1], color='red', label='Pred Class 1')
ax2.set_title(f"Model Predictions (Accuracy: {acc * 100:.1f}%)")
ax2.legend()

plt.show()