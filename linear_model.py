# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

# 1. Generate Dummy Data
# Random seed for reproducibility
np.random.seed(42)

# Generate random data for features (X) and target (y)
X = np.random.rand(100, 1) * 10  # 100 data points between 0 and 10
y = 2.5 * X.flatten() + np.random.randn(100) * 2  # y = 2.5x + noise

# 2. Split Data into Train and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Initialize and Train the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Evaluate the Model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"Model Coefficient: {model.coef_[0]:.2f}")
print(f"Model Intercept: {model.intercept_:.2f}")

# 5. Save the Model using Joblib
model_filename = 'linear_regression_model.joblib'
joblib.dump(model, model_filename)
print(f"Model saved as '{model_filename}'")

# # 6. Load the Saved Model and Test Prediction
# loaded_model = joblib.load(model_filename)
# sample_data = np.array([[5]])  # Example input for prediction
# prediction = loaded_model.predict(sample_data)
# print(f"Predicted value for input {sample_data.flatten()[0]}: {prediction[0]:.2f}")
