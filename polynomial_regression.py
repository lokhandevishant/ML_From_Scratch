import numpy as np
import matplotlib.pyplot as plt 

# Set seed for reproducibility
np.random.seed(50)

# Generate Data
X = np.random.rand(1000, 1)
# True function: y = 5x^2 + noise
y = 5 * (X**2) + np.random.rand(1000, 1)


class PolynomialRegression():
    def __init__(self, degree=2, learning_rate=0.01, iterations=1000):
        self.degree = degree
        self.lr = learning_rate
        self.iterations = iterations
        self.weight = None
        self.bias = None 

    def _create_polynomial_features(self, X):
        """
        Takes the original X and appends new columns for X^2, X^3, etc.
        based on the specified degree.
        """
        X_poly = X.copy()
        for d in range(2, self.degree + 1):
            X_poly = np.append(X_poly, X**d, axis=1)
        return X_poly

    def fit(self, X, y):
        # 1. Transform X to include polynomial features first
        X_poly = self._create_polynomial_features(X)
        
        n_samples, n_features = X_poly.shape 
        
        # Initialize weights (one for X, one for X^2, etc.) and bias
        self.weight = np.zeros((n_features, 1))
        self.bias = 0 
        
        # Ensure y is properly shaped as a column vector
        y = y.reshape(n_samples, 1)

        # Standard Batch Gradient Descent (just like your Linear Regression!)
        for _ in range(self.iterations):
            # y = Xw + b (but X now includes our polynomial features)
            y_predicted = np.dot(X_poly, self.weight) + self.bias 

            # Calculate Gradients
            dw = (1/n_samples) * np.dot(X_poly.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)

            # Update Parameters
            self.weight -= (self.lr * dw) 
            self.bias -= (self.lr * db) 

    def predict(self, X):
        # Must transform X the exact same way before predicting
        X_poly = self._create_polynomial_features(X)
        return np.dot(X_poly, self.weight) + self.bias
    
    def r2_score(self, y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)   # Sum of Squares of Residuals
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)   # Total Sum of Squares
        return 1 - (ss_res / ss_tot)


# --- Running the Model ---

# Instantiate and train (using standard Batch GD)
poly_model = PolynomialRegression(degree=2, learning_rate=0.1, iterations=2000)
poly_model.fit(X, y)

# Predict
y_hat = poly_model.predict(X)

# Calculate Accuracy
accuracy = poly_model.r2_score(y, y_hat)
print(f"R2 Score: {accuracy:.4f}")

# --- Plotting ---
fig = plt.figure(figsize=(8, 6))
plt.plot(X, y, 'y.', label="True Data Points")
plt.plot(X, y_hat, 'r.', label="Preds from Polynomial Regression")
plt.legend()
plt.xlabel('X - Input')
plt.ylabel('y - target / true')
plt.title('Polynomial Regression')
plt.show()