import numpy as np

class LassoRegression:
    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-4):
        """
        Initialize the Lasso Regression model.
        
        alpha: Regularization strength.
        max_iter: Maximum number of iterations for Coordinate Descent.
        tol: Tolerance for convergence checking.
        """
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None

    def _soft_threshold(self, rho, alpha):
        """
        Applies the soft thresholding mathematical operator.
        """
        if rho < -alpha:
            return rho + alpha
        elif rho > alpha:
            return rho - alpha
        else:
            # This is where coefficients get driven exactly to zero!
            return 0.0

    def fit(self, X, y):
        """
        Fit the model using Coordinate Descent.
        """
        n_samples, n_features = X.shape
        
        # Step 1: Add bias column (1s) to the feature matrix
        X_with_bias = np.c_[np.ones((n_samples, 1)), X]
        n_features_extended = n_features + 1
        
        # Step 2: Initialize weights to zeros
        self.weights = np.zeros(n_features_extended)
        
        # Step 3: Precompute the denominator z_j (sum of squared values for each feature)
        # We add a tiny epsilon to prevent division by zero in edge cases
        z = np.sum(X_with_bias**2, axis=0) + 1e-8 
        
        # Step 4: Coordinate Descent Loop
        for iteration in range(self.max_iter):
            weights_old = self.weights.copy()
            
            # Loop over every weight (including bias) one at a time
            for j in range(n_features_extended):
                
                # Calculate current predictions with all current weights
                y_pred = X_with_bias.dot(self.weights)
                
                # Calculate rho_j
                # We add back the contribution of the current feature j to isolate the residual 
                # as if feature j was completely removed from the model.
                rho_j = X_with_bias[:, j].T.dot(y - y_pred + self.weights[j] * X_with_bias[:, j])
                
                # Step 5: Update the weight
                if j == 0:
                    # We do NOT regularize the bias (intercept) term
                    self.weights[j] = rho_j / z[j]
                else:
                    # We apply the soft thresholding operator to the features
                    self.weights[j] = self._soft_threshold(rho_j, self.alpha) / z[j]
                    
            # Check for convergence (if weights stop changing significantly)
            if np.sum(np.abs(self.weights - weights_old)) < self.tol:
                break

    def predict(self, X):
        """
        Predict target values for new input data.
        """
        n_samples = X.shape[0]
        X_with_bias = np.c_[np.ones((n_samples, 1)), X]
        return X_with_bias.dot(self.weights)
    




# Generate synthetic data
np.random.seed(42)

# Feature 1: Meaningful data
X_real = np.random.rand(100, 1) * 10 
# Feature 2: Pure random noise
X_noise = np.random.rand(100, 1) * 10 

# Combine into our feature matrix X
X_train = np.hstack((X_real, X_noise))

# True equation relies ONLY on X_real: y = 4.5 * X_real + 10.0 + slight noise
y_train = 4.5 * X_real.squeeze() + 10.0 + np.random.randn(100) * 1.5

# Initialize and fit the custom Lasso Regression model
# We use a relatively high alpha to clearly see the feature selection
model = LassoRegression(alpha=50.0, iterations=2000)
model.fit(X_train, y_train)

# Extract parameters
intercept = model.weights[0]
coef_real = model.weights[1]
coef_noise = model.weights[2]

print("--- Lasso Model Results ---")
print(f"Learned Intercept:        {intercept:.4f} (Expected ~10.0)")
print(f"Learned Coef (Real):      {coef_real:.4f} (Expected ~4.5)")
print(f"Learned Coef (Noise):     {coef_noise:.4f} (Expected 0.0)")