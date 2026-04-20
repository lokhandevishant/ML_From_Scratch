import numpy as np

class LinearRegression():
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
            y_predicted = np.dot(X, self.weight)+ self.bias 

            #Calculate Gradients (Derivative) 
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)

            #Update Parameters
            self.weight = self.weight - (self.lr * dw) 
            self.bias = self.bias - (self.lr * db) 

    def predict(self, X):
        return np.dot(X, self.weight) + self.bias
    
    #Implementing R2 score to measure accuracy 
    def r2_score(self, y_true, y_pred):
        ss_res = np.sum((y_true-y_pred) ** 2)   #Sum of Squares of Residuals
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)   #Total Sum of Squares
        return 1 - (ss_res / ss_tot)
    
    def MSE(self, y_true, y_pred):
        mse = np.mean((y_true - y_pred) ** 2)
        return mse
    



#Testing the Algorithm
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


X, y = make_regression(n_samples=500, n_features=5, noise=15, random_state=42)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("First 5 rows of X_train:")
print(X_train[:5]) 

print("\nFirst 5 values of y_train:")
print(y_train[:5])

model = LinearRegression(learning_rate=0.1, iterations=2000)

print("Starting Training...")
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = model.r2_score(y_test, predictions)
mse = model.MSE(y_test, predictions)

print("-" * 30)
print(f"Final R2 Accuracy Score: {accuracy:.4f}")
print(f"MSE: {mse:.4f}")
print(f"Sample Prediction: {predictions[5]:.2f} (Actual: {y_test[5]:.2f})")

#Testing against the Scikit-learn model
from sklearn.linear_model import LinearRegression as SklearnLR
from sklearn.metrics import mean_squared_error

sk_model = SklearnLR()
sk_model.fit(X_train, y_train)
sk_y_pred = sk_model.predict(X_test)
print(f"Sklearn R2 Score: {sk_model.score(X_test, y_test):.4f}")
print(f"MSE: {mean_squared_error(y_test, sk_y_pred):.4f}")


'''
SAMPLE OUTPUT
Starting Training...
------------------------------
Final R2 Accuracy Score: 0.9800
MSE: 239.2313
Sample Prediction: 10.40 (Actual: -6.04)
Sklearn R2 Score: 0.9800
MSE: 239.2313
'''

import matplotlib.pyplot as plt
import seaborn as sns

# 1. Calculate the values
residuals = y_test - predictions

plt.figure(figsize=(10, 6))

# 2. Scatter plot of residuals
# Use 'lowess=True' to show the trend of the residuals
sns.regplot(x=predictions, y=residuals, lowess=True, 
            line_kws={'color': 'red', 'lw': 2}, 
            scatter_kws={'alpha': 0.4})

# 3. The Optimal Line (Horizontal at 0)
plt.axhline(y=0, color='black', linestyle='--')

plt.title('Residuals vs Fitted (Check for Randomness)')
plt.xlabel('Fitted Values (Predicted)')
plt.ylabel('Residuals (Actual - Predicted)')
plt.show()