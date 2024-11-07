import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]).reshape(-1, 1)
Y = np.array([10.06, 6.6, 10.91, 17.96, 18.47, 9.09, 18.8, 16.39, 18.59, 22.64, 
              23.58, 30.82, 30.04, 29.49, 32.78, 34.33, 40.98, 36.18, 
              40.25, 37.58])
n = len(X)
mean_X = np.sum(X) / n
variance_X = np.sum((X - mean_X) ** 2) / (n - 1)  

mean_Y = np.sum(Y) / n
variance_Y = np.sum((Y - mean_Y) ** 2) / (n - 1)  

covariance_XY = np.sum((X.flatten() - mean_X) * (Y - mean_Y)) / (n - 1)  

θ1 = covariance_XY / variance_X
θ0 = mean_Y - θ1 * mean_X

print(f"θ0: {θ0}  &  θ1: {θ1}")
print(f"Ecuación: {θ0} + {θ1} * X")

# Regresión lineal usando scikit-learn
model = LinearRegression()
model.fit(X, Y)

slope = model.coef_[0]
intercept = model.intercept_

print(f"θ0 (intercept): {intercept}")
print(f"θ1 (slope): {slope}")

# Análisis de regresión usando statsmodels
X_sm = sm.add_constant(X)  
model_sm = sm.OLS(Y, X_sm)
results = model_sm.fit()

print("\nResumen de la regresión usando statsmodels:")
print(results.summary())
