import numpy as np

class RidgeRegression:
    def __init__(self, alpha=1.0):
        """
        Initialize the Ridge Regression model.
        
        alpha: Regularization strength. Larger values specify stronger regularization.
        """
        self.alpha = alpha
        self.weights = None

    def fit(self, X, y):
        """
        Fit the model using the closed-form solution.
        """
        n_samples, n_features = X.shape

        # Step 1: Add a bias column (column of 1s) to the feature matrix X
        X_with_bias = np.c_[np.ones((n_samples, 1)), X]

        # Step 2: Create an identity matrix of size (n_features + 1)
        # It's +1 because we added the bias column.
        I = np.eye(n_features + 1)

        # Step 3: Set the first diagonal element to 0
        # This ensures we do not apply the L2 penalty to the intercept/bias term.
        I[0, 0] = 0

        # Step 4: Apply the closed-form formula: beta = (X^T * X + alpha * I)^-1 * X^T * y
        # Calculate the dot product of X transposed and X
        X_T_X = X_with_bias.T.dot(X_with_bias)
        
        # Add the regularization term
        penalty = self.alpha * I
        
        # Calculate the inverse of the left side
        inverse_term = np.linalg.inv(X_T_X + penalty)
        
        # Calculate X transposed dot y
        X_T_y = X_with_bias.T.dot(y)
        
        # Multiply the inverse term by X_T_y to get the final weights
        self.weights = inverse_term.dot(X_T_y)

    def predict(self, X):
        """
        Predict target values for new input data.
        """
        n_samples = X.shape[0]
        
        # Step 5: Add the bias column to the new data
        X_with_bias = np.c_[np.ones((n_samples, 1)), X]
        
        # Step 6: Compute predictions (y_hat = X * beta)
        return X_with_bias.dot(self.weights)
    

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

X, y = fetch_california_housing(return_X_y=True)

reg = RidgeRegression(alpha = 2)
reg.fit(X,y)
y_pred = reg.predict(X)
print(r2_score(y, y_pred))


regr = Ridge(alpha = 2)
regr.fit(X,y)
y_pred = regr.predict(X)
print(r2_score(y, y_pred))