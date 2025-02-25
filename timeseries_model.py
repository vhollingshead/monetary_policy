# Import necessary libraries
import streamlit as st
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import joblib
import matplotlib.pyplot as plt

# 1. Generate Dummy Time Series Data
np.random.seed(42)
date_range = pd.date_range(start='2020-01-01', periods=100, freq='D')
data = 50 + np.arange(100) * 0.5 + np.random.normal(0, 2, 100)
time_series_data = pd.Series(data, index=date_range)

# 2. Train ARIMA Model (if not already saved)
model = ARIMA(time_series_data, order=(1, 1, 1))
model_fit = model.fit()

# Save the model using joblib
model_filename = 'time_series_arima_model.joblib'
joblib.dump(model_fit, model_filename)