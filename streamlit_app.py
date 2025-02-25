import streamlit as st
import pandas as pd
import math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='Monetary Policy Dashboard_hellooo',
    page_icon=':earth_americas:', # This is an emoji shortcode. Could be a URL too.
)

# -----------------------------------------------------------------------------
# Declare some useful functions.

def main():
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", ["Product (MVP)","About", "Our Solution", "Home", "Use Case"])
    
    if selection == "Product (MVP)":
        demo()
    elif selection == "Home":
        home()
    elif selection == "About":
        about()
    elif selection == "Our Solution":
        our_solution()
    elif selection == "Use Case":
        use_case()

def demo():
    model_filename = 'linear_regression_model.joblib'
    loaded_model = joblib.load(model_filename)

    # Load the time series trained model
    timeseries_loaded_model = joblib.load(model_filename)


    # Generate dummy time series data
    def generate_data(interest_rate):
        np.random.seed(42)
        time = pd.date_range(start="2020-01-01", periods=50, freq='M')
        base_gini = 0.35 + (interest_rate * 0.005)  # Simulated effect of interest rate on Gini coefficient
        gini_values = base_gini + np.random.normal(0, 0.02, len(time))  # Adding some noise
        return pd.DataFrame({"Time": time, "Gini Coefficient": gini_values})

    # Streamlit UI
    st.title("Monetary Policy Dashboard")

    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Slider for interest rate
        interest_rate = st.slider("Interest Rate (%)", min_value=0.0, max_value=10.0, step=0.1, value=5.0)
        input_value = np.array([[interest_rate]])
        prediction = loaded_model.predict(input_value)
    
    with col2:
        # Display a random Gini coefficient
        calc_gini = round(prediction[0], 2)
        st.metric(label="Gini Coefficient Change", value=calc_gini)

    # 4. Adjust Time Series Based on Interest Rate
    # Simulating the effect of the interest rate on future values
    # Higher interest rates dampen growth in this example
    adjustment_factor = 1 - (interest_rate / 100)  # Reduces future forecast values based on interest rate

    # Generate forecast for the next 10 days
    forecast = timeseries_loaded_model.forecast(steps=10)
    adjusted_forecast = forecast * adjustment_factor  # Adjusting the forecast based on interest rate

    # Display the forecast
    st.write(f"Predicted Values Adjusted for an Interest Rate of {interest_rate}%:")
    st.write(adjusted_forecast.round(2))

    # 5. Plot Historical Data and Adjusted Forecast
    st.write("### Time Series Forecast Visualization")
    plt.figure(figsize=(10, 6))
    plt.plot(time_series_data, label='Historical Data')
    forecast_index = pd.date_range(start=time_series_data.index[-1] + pd.Timedelta(days=1), periods=10, freq='D')
    plt.plot(forecast_index, adjusted_forecast, label=f'Forecast (Adjusted for {interest_rate}%)', color='red')
    plt.title('Time Series Forecast Adjusted by Interest Rate')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()

    # Display the plot in Streamlit
    st.pyplot(plt)

    

def home():
    st.title("Home")
    st.write("Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.")

def about():
    st.title("About")
    st.write("Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.")

def our_solution():
    st.title("Our Solution")
    st.write("Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.")
    
    st.subheader("High Level Overview")
    st.write("Lorem ipsum dolor sit amet, consectetur adipiscing elit.")
    
    st.subheader("Methodology")
    st.write("Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.")
    
    st.subheader("Data pipeline")
    st.write("Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.")
    
    st.subheader("Training dataset (if applicable)")
    st.write("Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.")
    
    st.subheader("Data pre-processing")
    st.write("Curabitur pretium tincidunt lacus. Nulla gravida orci a odio.")
    
    st.subheader("Modelling")
    st.write("Nullam varius, turpis et commodo pharetra, est eros bibendum elit.")
    
    st.subheader("Model Evaluation")
    st.write("Donec odio tempus molestie, porttitor ut, iaculis quis, sem.")
    
    st.subheader("Key Learnings & Impact")
    st.write("Maecenas libero. Curabitur suscipit suscipit tellus.")
    
    st.subheader("Key Contributions")
    st.write("Sed lectus. Integer euismod lacus luctus magna.")
    
    st.subheader("Future Work")
    st.write("Vivamus quis mi. Phasellus a est.")
    
    st.subheader("Acknowledgements")
    st.write("Pellentesque dapibus hendrerit tortor. Praesent egestas tristique nibh.")

def use_case():
    st.title("Use Case")
    st.write("Curabitur pretium tincidunt lacus. Nulla gravida orci a odio. Nullam varius, turpis et commodo pharetra, est eros bibendum elit.")

if __name__ == "__main__":
    main()
