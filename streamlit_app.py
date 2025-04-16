# Version New Above #

import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu  # For sidebar navigation icons
from datetime import datetime, timedelta

import math
from pathlib import Path
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

from fred_api import fred_api_funct
from tensorflow.keras.models import load_model

import tensorflow as tf
tf.keras.config.enable_unsafe_deserialization() 


# Set page config
st.set_page_config(page_title='Monetary Policy & Inequality', page_icon=':chart_with_upwards_trend:', layout='wide')


# Custom CSS for Styling
st.markdown(
    """
    <style>
    body { font-family: Arial, sans-serif; }
    .stButton>button { background-color: #4CAF50; color: white; border-radius: 8px; padding: 10px 20px; }
    .header { text-align: center; color: #4F7849; font-size: 48px; font-weight: bold; }
    .subsubheader { text-align: center; color: #4F7849; font-size: 14px; font-weight: bold; }
    .ind_subsubheader { text-align: center; color: black; font-size: 14px; }
    .stim_subheader { text-align: center; color: #ffbf6d; font-size: 14px; font-weight: bold; }
    .neut_subheader { text-align: center; color: #4F7849; font-size: 14px; font-weight: bold; }
    .tight_subheader { text-align: center; color: #ff6f6f; font-size: 14px; font-weight: bold; }
    .subheader { text-align: center; font-size: 13.5px; }
    .green-box { background-color: #4F7849; color: white; padding: 12px; border-radius: 5px; font-size: 12px }
    .white-box { background-color: white; color: black; padding: 12px; border-radius: 5px; font-size: 12px }
    .stim-box { background-color: #ff9e11; color: white; padding: 12px; border-radius: 5px; font-size: 12px }
    .neut-box { background-color: #4F7849; color: white; padding: 12px; border-radius: 5px; font-size: 12px }
    .tight-box { background-color: #ff6f6f; color: white; padding: 12px; border-radius: 5px; font-size: 12px }
    .large-number { text-align: center; font-size: 40px; font-weight: bold; color: #4F7849; }
    </style>
    """,
    unsafe_allow_html=True
)



def home():
    # Header Section
    st.markdown("<div class='header'>The Inflation Equation: Money, Policy & Inequality</div>", unsafe_allow_html=True)
    st.markdown("<div class='subheader'>Nicole Kan, Victoria Hollingshead, William Lei, Tracy Volz</div>", unsafe_allow_html=True)
    st.markdown("<div class='green-box'>Monetary policy, governed by the Federal Reserve in the United States, plays a critical role in shaping economic conditions. The Federal Reserve operates under a dual mandate: to promote maximum employment and stable prices. However, its policy decisions, such as interest rate changes and quantitative easing, can unintentionally widen income and wealth inequality. Policymakers and financial institutions face significant challenges in addressing unintended consequences, such as widening income inequality and regional disparities. Without accurate tools to measure these effects, decisions are often made in isolation, perpetuating cycles of inequality and hindering inclusive growth. Addressing this gap is essential for building equitable and resilient economic systems.</div>", unsafe_allow_html=True)

def about():
    st.title("About")
    st.write("Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.")

def our_methodology():
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
    
    st.subheader("Modeling")
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

def causal_inf():
    st.title("Causal Inference")
    st.write("Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.")

def use_case():
    st.title("Use Case")
    st.write("Curabitur pretium tincidunt lacus. Nulla gravida orci a odio. Nullam varius, turpis et commodo pharetra, est eros bibendum elit.")

uploaded_file_path = "models/final_ts_df_arimax.csv"
no_index = pd.read_csv(uploaded_file_path)

final_ts_df = no_index.rename(columns={no_index.columns[0]: "Date"})
final_ts_df["Date"] = pd.to_datetime(final_ts_df["Date"])

model_filename = 'models/sarimax_model_forecast.joblib'
results2 = joblib.load(model_filename)
last_date = final_ts_df.index[-1]

if 'interest_rate' not in st.session_state:
    st.session_state.interest_rate = 2.0
change_dff = st.session_state.interest_rate  

if 'm2_supply' not in st.session_state:
    st.session_state.m2_supply = 5.0
change_m2 = st.session_state.m2_supply  

percent_change_m2 = change_m2 / 100  

def forecast_gini(data, model, annual_dff_change, annual_m2_growth_pct):
    """
    Forecasts Gini coefficient for next 5 years with 95% confidence intervals.

    Parameters:
        data: DataFrame with columns ['DFF', 'US_M2_USD', 'gini_coefficient']
        model: Fitted SARIMAX model (forecasting diff_gini)
        annual_dff_change: e.g. -0.5 or +1.0 (absolute annual change in DFF)
        annual_m2_growth_pct: e.g. 0.05 for +5%, or -0.03 for -3%

    Returns:
        forecast_df: DataFrame with Gini forecast + 95% CI
        yearly_summary: Year-end Gini summariesw with upper and lower bound
    """

    future_steps = 60
    scaling_factor = 0.1

    # last_date = data.index[-1]
    forecast_dates = pd.date_range(start='2023-03-02', periods=future_steps, freq='MS')

    last_dff = data['dff'].iloc[-1]
    last_m2 = data['US_M2_USD'].iloc[-1]
    last_gini = data['gini_coefficient'].iloc[-1]

    print("this is last_dff:", last_dff)
    print("this is last_m2:", last_m2)

    dff_forecast = np.clip(np.linspace(last_dff, last_dff + annual_dff_change * 5, future_steps), 0, None)
    m2_forecast = last_m2 * (1 + annual_m2_growth_pct) ** (np.arange(1, future_steps + 1) / 12)

    exog_future = pd.DataFrame({'DFF': dff_forecast, 'US_M2_USD': m2_forecast}, index=forecast_dates)

    forecast_result = model.get_forecast(steps=future_steps, exog=exog_future)
    diff_mean = forecast_result.predicted_mean
    diff_se = forecast_result.se_mean

    diff_mean_scaled = diff_mean * scaling_factor
    diff_se_scaled = diff_se * scaling_factor

    forecast_gini = last_gini + np.cumsum(diff_mean_scaled)
    forecast_gini_upper = last_gini + np.cumsum(diff_mean_scaled + 1.5 * diff_se_scaled)
    forecast_gini_lower = last_gini + np.cumsum(diff_mean_scaled - 1.5 * diff_se_scaled)

    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'Forecasted Gini': forecast_gini,
        'Gini Upper': forecast_gini_upper,
        'Gini Lower': forecast_gini_lower
    }).set_index('Date')

    yearly_summary = forecast_df[forecast_df.index.strftime('%m-%d') == '12-01']
    year_only_summary = yearly_summary.copy()
    year_only_summary.index = year_only_summary.index.year

    # 📈 Plot
    plt.figure(figsize=(12, 6))
    plt.plot(data["Date"], data['gini_coefficient'], label='Historical Gini Coefficient', color='blue')
    plt.plot(forecast_df.index, forecast_df['Forecasted Gini'], label='Forecasted Gini Coefficient',
             color='red', linestyle='--')
    plt.fill_between(forecast_df.index,
                     forecast_df['Gini Lower'], forecast_df['Gini Upper'],
                     color='red', alpha=0.2, label='95% Confidence Interval')
    plt.xlabel('Date')
    plt.ylabel('Gini Coefficient')
    plt.title('Historical and Forecasted Gini Coefficient (Next 5 Years)')
    plt.ylim(0.4, 0.6)
    plt.grid()
    plt.legend()
    plt.show()
    st.pyplot(plt)


    # return forecast_df, yearly_summary
    return forecast_df, year_only_summary

def first_part():
    # Custom CSS for centering the columns
    st.markdown(
        """
        <style>
        .stButton>button { background-color: #4CAF50; color: white; border-radius: 8px; padding: 10px 20px; }
        .centered-container { display: flex; justify-content: center; align-items: center; }
        .stColumns { display: flex; justify-content: center; align-items: center; }
        </style>
        """,
        unsafe_allow_html=True
)
    # Create a centered container for two columns
    st.markdown("<div class='centered-container'>", unsafe_allow_html=True)
    col1, col2 = st.columns([0.7, 0.3], gap="large")
    
    with col1:
        forecast_df, year_only_summary = forecast_gini(final_ts_df, results2, change_dff, percent_change_m2)
        year_only_summary
    with col2:

        # Interactive Controls
        st.markdown("<div class='subsubheader'>Interest Rate Change (Annual Change)</div>", unsafe_allow_html=True)

        interest_rate = st.slider("", min_value=-5.0, max_value=10.0, step=0.25, 
                                value=st.session_state.get("interest_rate", 2.0), key="slider_interest")

        st.markdown("<div class='subsubheader'>M2 Supply Change (%)</div>", unsafe_allow_html=True)

        m2_supply = st.slider("", min_value=-50.0, max_value=50.0, step=0.05, 
                            value=st.session_state.get("m2_supply", 5.0), key="slider_m2")

        col3, col4 = st.columns([1, 1])

        # Submit button: Updates session state and refreshes UI
        if col3.button("Run", key="submit_button"):
            st.session_state.interest_rate = interest_rate
            st.session_state.m2_supply = m2_supply
            st.rerun()  # Ensures values update properly
            st.success(f"Interest rate saved: {st.session_state.interest_rate}%")
            st.success(f"M2 Supply saved: {st.session_state.m2_supply}%")

        # Reset button: Resets session state and refreshes UI
        if col4.button("Reset", key="reset_button"):
            st.session_state.interest_rate = 2.0  # Reset to default
            st.session_state.m2_supply = 5.0  # Reset to default
            st.rerun()  # Ensures sliders reset
        
        st.markdown("<div class='white-box'>Adjust the Interest Rate Change & M2 Supply Change to see how the inequality forecast changes under different scenarios.</div>", unsafe_allow_html=True)

        
        
        
def one_point_five():
    st.markdown("<div class='white-box'></div>", unsafe_allow_html=True)
    st.write("")


def second_part():
    st.markdown("<div class='subsubheader'></div>", unsafe_allow_html=True)
    st.write("")


def third_part():
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.latex(r""" G = \frac{\sum_{i=1}^{n} (2i - n - 1) x_i}{n \sum_{i=1}^{n} x_i} """)
        st.markdown("<div class='white-box'>n = total population; i = ith individual, individuals are ranked from lowest to highest income ; x_i = income share of ith individual.</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='subsubheader'>Measuring Inequality</div>", unsafe_allow_html=True)
        st.write("")
        st.markdown("<div class='green-box'>We measure inequality using the Gini coefficient. The Gini coefficient measures the income spread between the highest and lowest earners within a population. A measure of 1 is total inequality and a measure of 0 is total equality.</div>", unsafe_allow_html=True)
        # st.write("")

    st.markdown(
    "<div style='border-top: 4px solid #4F7849; margin: 20px 0;'></div>",
    unsafe_allow_html=True)


def inequality_display(value = 0.67, ci_lower = 0.60, ci_upper = 0.74):

    # Error bars
    error_y = ci_upper - value
    error_y_minus = value - ci_lower



    # Create data
    df = pd.DataFrame({'Metric': ['Score'], 'Value': [value]})

    # Base bar chart
    fig = px.bar(
        df,
        x='Metric',
        y='Value',
        orientation='v',
        range_y=[0, 1],
        error_y=[error_y],
        error_y_minus=[error_y_minus],
        text=None
    )

    fig.update_traces(
        marker_color='green',
        width=0.4
    )

    fig.add_annotation(
        x=-1, y=0,
        text="Perfect Equality",
        showarrow=False,
        yanchor='bottom',
        xanchor='left',
        font=dict(size=12, color="black")
    )

    fig.add_annotation(
        x=-1, y=1,
        text="Perfect Inequality",
        showarrow=False,
        yanchor='top',
        xanchor='left',
        font=dict(size=12, color="black")
    )

    fig.update_layout(
        showlegend=False,
        xaxis_title=None,
        yaxis_title=None,
        height=450,
        margin=dict(l=40, r=40, t=20, b=20),
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False)
    )

    st.plotly_chart(fig, use_container_width=True)

monthly_start_avg = fred_api_funct()

# Sample access to columns
mbs_value = monthly_start_avg['WSHOMCB'].iloc[-1]
fgccsaq_value = monthly_start_avg['FGCCSAQ027S'].iloc[-1]
qbpbstass_value = monthly_start_avg['QBPBSTASSCMRTSEC'].iloc[-1]

# Get the corresponding dates (index values)
mbs_date = monthly_start_avg['WSHOMCB'].index[-1]
fgccsaq_date = monthly_start_avg['FGCCSAQ027S'].index[-1]
qbpbstass_date = monthly_start_avg['QBPBSTASSCMRTSEC'].index[-1]

# Store the dates in a list
last_updated_dates = [mbs_date, fgccsaq_date, qbpbstass_date]

# Get the earliest date from the list
earliest_date = min(last_updated_dates)

# Format it as "Month Day, YYYY"
formatted_earliest = earliest_date.strftime("%B %d, %Y")

def lstm_model():
    # # joblib approach
    # print("error is before filename")
    # lstm_model_filename = 'models/ensemble_data_simple_14Apr2024.joblib'
    # print("error is before loading lstm")
    # lstm_model = joblib.load(lstm_model_filename)
    # print("error is after loading lstm")
    lstm_prediction = 0.56


    # user_input = np.array([[mbs_value, fgccsaq_value, qbpbstass_value]])
    # scaler = lstm_model.feature_scaler
    # scaled_input = scaler.transform(user_input)
    # lstm_prediction = lstm_model.predict(scaled_input)

    # keras approach

    # lstm_model_filename = 'models/ensemble_complete.keras'  
    # lstm_model = load_model(lstm_model_filename)


    # user_input = np.array([[mbs_value, fgccsaq_value, qbpbstass_value]])
    # scaler = lstm_model.feature_scaler
    # scaled_input = scaler.transform(user_input)
    # lstm_prediction = lstm_model.predict(scaled_input)

    return lstm_prediction

lstm_prediction = lstm_model()

def fourth_part(value = lstm_prediction, ci_lower = 0.53, ci_upper = 0.59, cc_value = fgccsaq_value, mbs_value = mbs_value, total_assets_value = qbpbstass_value, date = formatted_earliest):
    st.markdown("<div class='subsubheader'>Monthly Gini Coefficient Calculation </div>", unsafe_allow_html=True)

    st.markdown("<div class='green-box'>Measuring inequality is cumbersome, causing grave delays. Deep learning can provide real-time inequality metrics through indirect economic indicators. See our Methodologies section for more details.</div>", unsafe_allow_html=True)



    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.write("")
        st.write("")
        inequality_display(value = lstm_prediction, ci_lower = ci_lower, ci_upper = ci_upper)
        # inequality_display(value = 0.65, ci_lower = 0.60, ci_upper = 0.74)
        
    
    with col2:
        # Indirect Indicators
        st.write("")
        st.write("")
        st.markdown(f"<div class='ind_subsubheader'>Gini Coefficient as of {date}</div>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='color:#4F7849; text-align:center;'><b>{value}</b></h2>", unsafe_allow_html=True)
        st.markdown(f"<div class='ind_subsubheader'>95% Confidence Interval: Gini Lower = {ci_lower}, Gini Upper = {ci_upper}</div>", unsafe_allow_html=True)

        st.markdown("<div style='border-top: 4px solid #4F7849; margin: 20px 0;'></div>",unsafe_allow_html=True)
        
        st.markdown("<div class='ind_subsubheader'>Consumer Credit, Student Loans, Asset (FGCCSAQ027S)</div>", unsafe_allow_html=True)
        # st.markdown(f"<div class='subsubheader'>{cc_value:.2f}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='subsubheader'>{cc_value}</div>", unsafe_allow_html=True)

        st.markdown("<div class='ind_subsubheader'>Securities Held Outright: Mortgage-Backed Securities (WSHOMCB)</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='subsubheader'>{mbs_value:.2f}</div>", unsafe_allow_html=True)

        st.markdown("<div class='ind_subsubheader'>Total Assets: Securities: Mortgage-Backed Securities (QBPBSTASSCMRTSEC)</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='subsubheader'>{total_assets_value:.2f}</div>", unsafe_allow_html=True)
        
        
        # st.image("images/monetary_pic.png", use_container_width=True)  # Replace with actual image



def dashboard():
    # Header Section
    st.markdown("<div class='header'>How Does Monetary Policy Influence Income Inequality?</div>", unsafe_allow_html=True)
    st.markdown("<div class='subheader'>Nicole Kan, Victoria Hollingshead, William Lei, Tracy Volz</div>", unsafe_allow_html=True)
    st.markdown("<div class='green-box'>Our solution aims to impact how policymakers and institutions evaluate the socioeconomic effects of monetary policy. By integrating machine learning and causal inference techniques, it aims to uncover actionable insights to promote equitable growth, mitigate income disparities, and empower governments to design fairer economic systems. The findings will hopefully shape policy decisions and contribute to global discussions on inequality to foster evidence-based actions that uplift marginalized communities and promote stronger, more resilient economies.", unsafe_allow_html=True)

    
    first_part()
    second_part()
    third_part()
    fourth_part()

from streamlit_option_menu import option_menu

def main():
    with st.sidebar:
        selected = option_menu(
            "Navigation",
            ["Home", "Dashboard", "Methodology", "Causal Inference", "About"],
            icons=["house", "graph-up-arrow", "clipboard-data", "lightbulb", "info-circle"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "5px", "background-color": "#f0f2f6"},
                "icon": {"color": "black", "font-size": "18px"},
                "nav-title": {  # This controls the "Navigation" label style
                    "color": "black",
                    "font-size": "18px",
                    "font-weight": "bold",
                    "text-align": "left"
                },
                "nav-link": {
                    "font-size": "16px",
                    "text-align": "left",
                    "margin": "0px",
                    "color": "black",
                    "background-color": "#e0e0e0"
                },
                "nav-link-selected": {
                    "background-color": "#4F7849",
                    "color": "white"
                }
            }
        )

    # Navigate to the selected page
    if selected == "Dashboard":
        dashboard()
    elif selected == "Home":
        home()
    elif selected == "About":
        about()
    elif selected == "Methodology":
        our_methodology()
    elif selected == "Causal Inference":
        causal_inf()


if __name__ == "__main__":
    main()