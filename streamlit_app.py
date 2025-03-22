# Version New Above #

import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu  # For sidebar navigation icons

import math
from pathlib import Path
import numpy as np
import joblib

# Set page config
st.set_page_config(page_title='Monetary Policy & Inequality', page_icon=':chart_with_upwards_trend:', layout='wide')

# Custom CSS for Styling
st.markdown(
    """
    <style>
    body { font-family: Arial, sans-serif; }
    .stButton>button { background-color: #4CAF50; color: white; border-radius: 8px; padding: 10px 20px; }
    .header { text-align: center; color: #4F7849; font-size: 48px; font-weight: bold; }
    .subheader { text-align: center; font-size: 12px; }
    .green-box { background-color: #4F7849; color: white; padding: 12px; border-radius: 5px; font-size: 10px }
    .large-number { font-size: 40px; font-weight: bold; color: #4F7849; }
    </style>
    """,
    unsafe_allow_html=True
)

# Header Section
st.markdown("<div class='header'>How Does Monetary Policy Influence Income Inequality?</div>", unsafe_allow_html=True)
st.markdown("<div class='subheader'>Nicole Kan, Victoria Hollingshead, William Lei, Tracy Volz</div>", unsafe_allow_html=True)
st.markdown("<div class='green-box'>Monetary policy, governed by the Federal Reserve in the United States, plays a critical role in shaping economic conditions. The Federal Reserve operates under a dual mandate: to promote maximum employment and stable prices. However, its policy decisions, such as interest rate changes and quantitative easing, can unintentionally widen income and wealth inequality. Policymakers and financial institutions face significant challenges in addressing unintended consequences, such as widening income inequality and regional disparities. Without accurate tools to measure these effects, decisions are often made in isolation, perpetuating cycles of inequality and hindering inclusive growth. Addressing this gap is essential for building equitable and resilient economic systems.</div>", unsafe_allow_html=True)

def main():
    st.sidebar.title("Navigation")
    with st.sidebar:
        selected = option_menu(
            "Navigation", ["Home", "Dashboard", "Our Solution", "About"],
            icons=["house", "graph-up-arrow", "lightbulb", "info-circle"],
            menu_icon="cast", default_index=1
        )
    
    if selected == "Dashboard":
        demo()
    elif selected == "Home":
        home()
    elif selected == "About":
        about()
    elif selected == "Our Solution":
        our_solution()

# # Sidebar Navigation with Icons
# with st.sidebar:
#     selected = option_menu(
#         "Navigation", ["Home", "Dashboard", "Policy Impact", "Gini Coefficient", "About"],
#         icons=["house", "graph-up-arrow", "lightbulb", "clipboard-data", "info-circle"],
#         menu_icon="cast", default_index=1
#     )

def first_part():
    # Create two columns
    col1, col2 = st.columns(2)

    with col1:
        # Interactive Controls
        st.subheader("Adjust Monetary Parameters")
        interest_rate = st.slider("Interest Rate (%)", min_value=0.0, max_value=10.0, step=0.1, value=5.0)
        m2_supply = st.slider("M2 Supply (Trillions)", min_value=0.0, max_value=20.0, step=0.1, value=10.0)

        col3, col4 = st.columns([1,1])
        col3.button("Submit", key="submit_button")
        col4.button("Reset", key="reset_button")

    with col2:
        st.subheader("Column 2")
        st.write("Duis euismod risus et mauris gravida, a malesuada mi efficitur. Sed dictum, felis at feugiat suscipit, nulla eros tincidunt ligula, a interdum tortor ligula id turpis.")
        st.write("Nam eu sem nec nulla malesuada fermentum. Quisque imperdiet nulla vel libero viverra, et fermentum arcu congue.")




def time_series_plot():
    model_filename = 'linear_regression_model.joblib'
    loaded_model = joblib.load(model_filename)

    # Load the time series trained model
    timeseries_model_filename = "time_series_arima_model.joblib"
    timeseries_loaded_model = joblib.load(timeseries_model_filename)


    # Generate dummy time series data
    def generate_data(interest_rate):
        np.random.seed(42)
        time = pd.date_range(start="2020-01-01", periods=50, freq='M')
        base_gini = 0.35 + (interest_rate * 0.005)  # Simulated effect of interest rate on Gini coefficient
        gini_values = base_gini + np.random.normal(0, 0.02, len(time))  # Adding some noise
        return pd.DataFrame({"Time": time, "Gini Coefficient": gini_values})

    # Streamlit UI
    st.title("Monetary Policy Dashboard")

    # 4. Adjust Time Series Based on Interest Rate
    # Simulating the effect of the interest rate on future values
    # Higher interest rates dampen growth in this example
    adjustment_factor = 1 - (interest_rate / 100)  # Reduces future forecast values based on interest rate

    # Generate forecast for the next 10 days
    forecast = timeseries_loaded_model.forecast(steps=10)
    adjusted_forecast = forecast * adjustment_factor  # Adjusting the forecast based on interest rate

    # Display the forecast
    st.write(f"Predicted Values Adjusted for an Interest Rate of {interest_rate}%:")
    # st.write(adjusted_forecast.round(2))

    # 1. Generate Dummy Time Series Data
    np.random.seed(42)
    date_range = pd.date_range(start='2020-01-01', periods=100, freq='D')
    data = 50 + np.arange(100) * 0.5 + np.random.normal(0, 2, 100)
    time_series_data = pd.Series(data, index=date_range)

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

def gini_coefficient():
    st.title("Measuring Inequality: Gini Coefficient")
    
    st.latex(r""" G = \frac{\sum_{i=1}^{n} (2i - n - 1) x_i}{n \sum_{i=1}^{n} x_i} """)
    
    st.write("The Gini coefficient measures income inequality. A score of 1 is total inequality, while 0 represents total equality.")
    # st.image("gini_formula.png", use_container_width=True)  # Replace with actual image
    
    st.subheader("Inequality Pulse Check")
    st.markdown("<div class='large-number'>2.95</div>", unsafe_allow_html=True)
    st.write("Real-time deep learning estimation of inequality based on indirect economic indicators.")
    
    # st.image("economic_indicators.png", use_container_width=True)  # Replace with actual image

# Placeholder for content sections
def dashboard():

    first_part()
    
    # Time Series Forecast Plot
    st.subheader("Time Series Plot")
    time_series_plot()
    # st.image("time_series_plot.png", use_container_width=True)  # Replace with actual image
    
    # Policy Impact Table
    st.subheader("Interest Rate & M2 Supply Impact")
    policy_data = pd.DataFrame({
        "Policy": ["Tightening (Stimulus)", "Neutral (Green)", "Stimulus (Orange)"],
        "Rates": ["Higher, Slower M2 Growth", "Stable Rates, Stable M2", "Lower Rates, Fast M2 Growth"],
        "Inequality Impact": ["Decreases Sharply", "Decreases Moderately", "Increases Sharply"]
    })
    st.table(policy_data)

    gini_coefficient()

if __name__ == "__main__":
    dashboard()

#  
# # Example API call to FRED for economic data
# def fetch_fred_data(series_id, api_key):
#     url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={api_key}&file_type=json"
#     response = requests.get(url)
#     if response.status_code == 200:
#         data = response.json()["observations"]
#         df = pd.DataFrame(data)
#         df["value"] = pd.to_numeric(df["value"], errors='coerce')
#         df["date"] = pd.to_datetime(df["date"])
#         return df
#     else:
#         st.error("Failed to fetch data from FRED API.")
#         return None








# Version 3/21 Below # 


# # Set the title and favicon that appear in the Browser's tab bar.
# st.set_page_config(
#     page_title='Monetary Policy Dashboard_hellooo',
#     page_icon=':earth_americas:', # This is an emoji shortcode. Could be a URL too.
# )

# # -----------------------------------------------------------------------------
# # Declare some useful functions.

# def main():
#     st.sidebar.title("Navigation")
#     selection = st.sidebar.radio("Go to", ["Product (MVP)","About", "Our Solution", "Home", "Use Case"])
    
#     if selection == "Product (MVP)":
#         demo()
#     elif selection == "Home":
#         home()
#     elif selection == "About":
#         about()
#     elif selection == "Our Solution":
#         our_solution()
#     elif selection == "Use Case":
#         use_case()



    

# def home():
#     st.title("Home")
#     st.write("Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.")

# def about():
#     st.title("About")
#     st.write("Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.")

# def our_solution():
#     st.title("Our Solution")
#     st.write("Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.")
    
#     st.subheader("High Level Overview")
#     st.write("Lorem ipsum dolor sit amet, consectetur adipiscing elit.")
    
#     st.subheader("Methodology")
#     st.write("Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.")
    
#     st.subheader("Data pipeline")
#     st.write("Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.")
    
#     st.subheader("Training dataset (if applicable)")
#     st.write("Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.")
    
#     st.subheader("Data pre-processing")
#     st.write("Curabitur pretium tincidunt lacus. Nulla gravida orci a odio.")
    
#     st.subheader("Modelling")
#     st.write("Nullam varius, turpis et commodo pharetra, est eros bibendum elit.")
    
#     st.subheader("Model Evaluation")
#     st.write("Donec odio tempus molestie, porttitor ut, iaculis quis, sem.")
    
#     st.subheader("Key Learnings & Impact")
#     st.write("Maecenas libero. Curabitur suscipit suscipit tellus.")
    
#     st.subheader("Key Contributions")
#     st.write("Sed lectus. Integer euismod lacus luctus magna.")
    
#     st.subheader("Future Work")
#     st.write("Vivamus quis mi. Phasellus a est.")
    
#     st.subheader("Acknowledgements")
#     st.write("Pellentesque dapibus hendrerit tortor. Praesent egestas tristique nibh.")

# def use_case():
#     st.title("Use Case")
#     st.write("Curabitur pretium tincidunt lacus. Nulla gravida orci a odio. Nullam varius, turpis et commodo pharetra, est eros bibendum elit.")

# if __name__ == "__main__":
#     main()
