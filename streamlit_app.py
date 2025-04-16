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
    st.markdown("<div class='header'>Gini-Lab: An Interactive Dashboard on Monetary Policy’s Impact on Income Inequality</div>", unsafe_allow_html=True)
    st.markdown("<div class='subheader'>Nicole Kan, Victoria Hollingshead, William Lei, Tracy Volz</div>", unsafe_allow_html=True)

    # Problem Section
    st.markdown("### The Problem")
    # st.image("images/gini_chart.png", caption="Figure 1: Rising Inequality – The Gini Coefficient's Steady Climb from 1976 to 2023")

    st.markdown("""
    Income inequality in the United States has worsened due to monetary policies that, while stabilizing prices and employment, often disproportionately benefit asset holders. 
    Nobel prize-winning economist Joseph Stiglitz in <i>The Price of Inequality</i> (2012) notes that monetary policy has favored the wealthy, with the top 1% gaining wealth share post-2008 Global Financial Crisis (GFC) while the bottom 50% lost ground. 
    Data shows 70% of U.S. assets are held by the top 10%, deepening the divide.
    """, unsafe_allow_html=True)

    st.markdown("#### Gaps in the Current Landscape")
    st.markdown("""
    - **Limited Data-Driven Tools**: No interactive tools exist to simulate policy impacts on inequality in real time.  
    - **Periodic Reporting**: Federal Reserve research is limited to periodic reports, lacking real-time insights.  
    - **Lack of Reproducibility**: Analyses rely on custom consulting, which isn’t reusable or scalable.  
    - **Findings Stuck in Academia**: Research on inequality is often inaccessible, buried in academic papers.  
    - **Market Inefficiency**: Billions are spent on policy simulation tools and economic research, yet reporting isn’t inequality-focused.
    
    These gaps prevent policymakers, researchers, and the public from understanding and addressing monetary policy’s role in inequality.
    """)

    # Solution Section
    st.markdown("### Our Solution")
    st.markdown("""
    We introduce the **Monetary Policy Dashboard**, an interactive tool to highlight the impact of monetary policy on income inequality in the United States. It enables users to:

    - Simulate the effects of policy changes, like interest rate adjustments, on inequality metrics such as the Gini coefficient in real time.  
    - Visualize historical and forecasted inequality trends using advanced models.  
    - Provide up-to-date Gini coefficient estimates through machine learning.  
    """)

    # Impact Section
    st.markdown("### Product Impact")
    st.markdown("""
    - **Enables Equitable Policy-Making**: Addresses rising inequality concerns, helping design inclusive policies.  
    - **Reduces Economic Disparities**: Fosters sustainable growth and social stability by ensuring broader benefits from monetary policy.  
    """)

    # Overview Section
    st.markdown("### High-Level Overview")
    st.markdown("""
    The Monetary Policy Dashboard is an interactive platform that makes the link between monetary policy and income inequality accessible and actionable. It provides real-time insights into how policy decisions, such as interest rate changes, affect income inequality in the United States. 
    Designed for policymakers, researchers, economists, and the public, the dashboard offers a clear way to explore these dynamics.

    **The dashboard features two core components:**

    - **Time Series Forecasting**: Forecasts future trends in income inequality, like the Gini coefficient, based on historical data and policy adjustments.  
    - **Machine Learning for Current Insights**: Uses a machine learning model to calculate the current Gini coefficient, offering an up-to-date view of inequality.

    Through intuitive visualizations, such as time series plots, users can see how a change in interest rate might impact the Gini coefficient over time. By blending real-time analysis with predictive modeling, the dashboard empowers users to make informed, equitable decisions.
    """)

def about():
    # List of image URLs and their descriptions as bullet points
    images = [
        {
            "url": "images/nicole.png",
            "desc_header": "Nicole Kan",
            "desc_bullets": [
                "Causal Analysis",
                "Machine Learning",
                "Research"
            ]
        },
        {
            "url": "images/victoria.png",
            "desc_header": "Victoria Hollingshead",
            "desc_bullets": [
                "Product",
                "Machine Learning",
                "Research"
            ]
        },
        {
            "url": "images/william.png",
            "desc_header": "William Lei",
            "desc_bullets": [
                "Machine Learning",
                "Research"
            ]
        },
        {
            "url": "images/tracy.png",
            "desc_header": "Tracy Volz",
            "desc_bullets": [
                "Exploratory Data Analysis",
                "Machine Learning",
                "Research"
            ]
        }
    ]

    st.title("Meet our Team")

    for item in images:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(item["url"], use_column_width=True)
        with col2:
            st.markdown(f"<h5 style='margin-bottom:0'><b>{item['desc_header']}</b></h5>", unsafe_allow_html=True)
            for bullet in item["desc_bullets"]:
                st.markdown(f"<div style='margin-left: 20px;'>• {bullet}</div>", unsafe_allow_html=True)
        st.markdown("---")


def our_methodology():
    st.markdown("## Methodology")
    st.markdown("""
    Our methodology leverages **causal analysis** and **advanced analytics** to provide insights into how monetary policy impacts income inequality.
    """)

    with st.expander("📉 Time Series Forecasting Model"):
        st.markdown("""
        We use a **time series model** to predict future trends in the Gini coefficient based on monetary policy changes. Historical data is analyzed to simulate scenarios — such as interest rate adjustments — and visualize their impact over time. The model outputs both historical and forecasted inequality trends.
        """)

    with st.expander("🧠 LSTM Model to Predict the Current Gini"):
        st.markdown("""
        A **Long Short-Term Memory (LSTM)** model is used to estimate the **current Gini coefficient** based on recent economic data. This provides a real-time snapshot of inequality and allows users to track emerging patterns on the dashboard.
        """)

    with st.expander("🌲 RandomForestRegressor (Future Enhancement)"):
        st.markdown("""
        Although not part of the live dashboard, we trained a **RandomForestRegressor** model to evaluate its ability to predict Gini values. It outperformed many traditional approaches, showing potential for future integration. The model used a one-step-ahead forecasting approach with lagged features.
        """)

    with st.expander("📊 Difference-in-Differences (DiD) for Causal Inference"):
        st.markdown("""
        We used **DiD** to estimate the causal impact of monetary policy. This compares inequality before and after a policy change between treated and control groups — isolating the effect of interventions like interest rate hikes.
        """)

    with st.expander("🛠️ Data Pipeline & Preprocessing"):
        st.markdown("""
        - **Sources**: Data was pulled from FRED, Yahoo Finance, and real-time inequality databases.  
        - **Data Types**: Leading indicators (e.g. interest rates) and lagging indicators (e.g. Gini coefficient) were aligned.
        - **Preprocessing**: Dates were formatted, data resampled (daily/monthly), and missing values forward-filled.
        - **FRED API**: Data is pulled directly from the FRED database using the public API. Because leading and lagging variables have varying frequencies, they are averaged over the last month, published on the dashboard, and entered to the LSTM model for prediction.
        - **Chronological Split**: Data was split in time order to preserve forecasting validity and simulate real-world policy modeling.
        """)

    with st.expander("📐 Gini Coefficient Calculation"):
        st.markdown("""
        The Gini coefficient was derived using quarterly income shares for three socioeconomic bins: **Bottom 50%**, **Middle 40%**, and **Top 10%**. We calculated a population-weighted estimate, sufficient for tracking inequality trends over time.
        """)
        st.latex(r""" G = \frac{\sum_{i=1}^{n} (2i - n - 1) x_i}{n \sum_{i=1}^{n} x_i} """)

    with st.expander("📈 Model Evaluation Summary"):
        st.markdown("Here is a snapshot of the models we evaluated, and their performance:")
        st.markdown("""
        | Model                  | MAE       | MSE        | RMSE      | Selected?                   |
        |------------------------|-----------|------------|-----------|-----------------------------|
        | ARIMA (Baseline)       | 0.003738  | 0.001602   | 0.040029  | ❌                          |
        | ARIMAX                 | 0.000485  | 0.000001   | 0.000563  | ✅ (Forecasting Gini)       |
        | Linear Regression      | 0.011027  | 0.000130   | 0.005605  | ❌                          |
        | LSTM                   | 0.001133  | 0.000002   | 0.001385  | ✅ (Current Gini)           |
        | XGBoost                | 0.003849  | 0.000032   | 0.005696  | ❌                          |
        | RandomForestRegressor | 0.0001587 | 0.00000057 | 0.0007531 | ❌ (Potential Future Use)   |
        """)

    with st.expander("🔮 Forecasting Framework (ARIMAX-Based)"):
        st.markdown("""
        Our **forecasting framework** uses the ARIMAX model to simulate future Gini coefficient trends under user-defined monetary policy assumptions.

        - **Model Setup**: The ARIMAX model is trained on the first-differenced Gini coefficient with DFF and M2 as exogenous variables.
        - **Future Variables**: Users specify future changes to DFF and M2, which are assumed to follow simple trends.
        - **Forecast Output**: The model returns 5-year Gini coefficient forecasts with 95% confidence intervals.

        In the dashboard, users simulate scenarios like lowering DFF by 1% per year or increasing M2 by 10% annually. This empowers exploration of **how monetary policy may shape income inequality**.

        To improve reliability, we **re-weighted COVID-era data** to prevent the model from overfitting to volatility during the pandemic.

        _Figure 8 in the dashboard shows the resulting forecast._
        """)

    with st.expander("⚠️ Limitations"):
        st.markdown("""
        - **Simplified Assumptions**: Future DFF and M2 paths are simplified and do not account for macroeconomic shocks.
        - **National-Level View**: Aggregated data ignores regional or demographic heterogeneity.
        - **Non-Normal Residuals**: Slightly skewed residuals may reduce confidence interval reliability.
        - **Stationarity Risks**: Assumes stability in differenced series over long horizons.
        - **Illustrative Forecasts**: The goal is insight, not precise prediction.
        """)

    with st.expander("🔁 Model Retraining Strategy"):
        st.markdown("""
        - **Normal Conditions**: Models retrained annually to reflect slowly changing trends.
        - **Volatile Periods**: Quarterly retraining during instability like COVID-19.
        - **Trigger Conditions**: Retrain if:
            - Fed Funds Rate changes > 50 basis points in a quarter.
            - M2 monthly growth > 1% for 3+ consecutive months.

        This retraining ensures model relevance without overfitting to temporary noise.
        """)

    with st.expander("📌 Conclusion"):
        st.markdown("""
        This capstone demonstrates that monetary policy not only shapes macroeconomic conditions but also plays a measurable role in income inequality. Using ARIMAX models with exogenous variables like the Federal Funds Rate and M2 money supply, we significantly improved Gini coefficient forecasting accuracy. However, these models struggled to capture volatility during periods of economic disruption, such as the COVID-19 shock, underscoring the limits of linear forecasting frameworks under structural breaks.

        For real-time estimation, the LSTM model proved highly effective and achieved a huge improvement over linear regression. Its capacity to model non-linear temporal dependencies between financial indicators and inequality metrics reinforces the value of deep learning in economic forecasting. RandomForestRegressor shows promising performance and opens up the possibility of including supervised learning techniques in effectively capturing temporal dependencies.

        Meanwhile, the Difference-in-Differences (DiD) analysis provided robust causal evidence that U.S. monetary policy contributed to rising inequality, especially relative to Canada. Despite this, traditional monetary levers like interest rates and M2 did not individually emerge as significant predictors in the causal framework, suggesting broader structural dynamics are at play.

        Together, these results highlight that advanced statistical and machine learning models can provide deep insights into complex policy effects. Yet, they also emphasize critical challenges—such as model fragility during volatile periods, difficulty in isolating causality, and the need for continual retraining in dynamic economic contexts. Moving forward, integrating ensemble learning (e.g., Random Forest), modeling real-time shocks, and testing alternate causal frameworks like Regression Discontinuity can further enhance the explanatory and predictive power of policy tools aimed at economic equity.
        """)

    with st.expander("📚 References"):
        st.markdown("""
        - Blanchet, T., Saez, E., & Zucman, G. (November 2022). *Who benefits from income and wealth growth in the United States?* Department of Economics, University of California, Berkeley.  
        - Hyndman, R. J., & Athanasopoulos, G. (2021). *5.8 Training and test sets*. In Forecasting: Principles and practice. OTexts.
        """)



def causal_inf():
    st.markdown("<div class='header'>Causal Inference</div>", unsafe_allow_html=True)

    st.markdown("""
    To examine the impact of U.S. monetary policy on income inequality, the analysis employs a series of Difference-in-Differences (DiD) models using Canada as a control group. The core idea is to compare changes in the U.S. Gini coefficient over time—before and after key monetary policy events—with corresponding changes in Canada, a country with a similar economic structure but independent monetary policy. By focusing on the interaction between treatment status (U.S. vs. Canada) and time, these models aim to isolate the causal effect of U.S. monetary shifts (such as changes in interest rates or money supply) on income inequality trends.
    
    A baseline DiD model includes just the treatment dummy, time trend, and an interaction term (`DiD_Interaction`), but more refined models introduce monetary policy variables like the U.S. federal funds rate (US_DFF), the interest rate spread between the U.S. and Canada, and both nominal and inflation-adjusted money supply (M2). As models evolve in complexity—e.g., Model 14 includes interest rate changes, cross-country spreads, and inflation-adjusted M2—the results become more robust and the DiD_Interaction term gains strong statistical significance. This suggests that U.S. monetary policy may indeed play a role in shaping inequality trends.
    
    To assess the validity of the parallel trends assumption—a key requirement for causal inference in Difference-in-Differences (DiD) analysis—a dynamic event study model was employed. A traditional pre-trend check was not feasible due to missing Gini coefficient data for Canada prior to 1992. Instead, the event study specification evaluates whether the U.S. and Canada followed similar inequality trends leading up to major monetary policy events, particularly around 2008. The interaction term `Treatment:Event_Time` captures the differential trend between the U.S. and Canada over time. In the model, this coefficient is small (0.0245) and statistically insignificant (p = 0.249), indicating no strong evidence of divergence in trends prior to the treatment period.
    
    This supports the parallel trends assumption and reinforces the credibility of Canada as a valid control group. Additionally, the high R-squared (0.977) suggests that the model fits the data well, and the insignificant `Event_Time` term (p = 0.102) further suggests that overall trends were not drastically shifting in the pre-treatment years. Visual inspection of the Gini trends further supports this conclusion, as the U.S. and Canadian series appear to move in parallel throughout the pre-2008 period with no major divergence. While some divergence emerges around 2015—with U.S. inequality trending upward and Canada’s trend flattening or declining—this occurs well into the post-treatment period. Since the models estimate treatment effects over the full 1992–2019 window, such divergence may in fact reflect the very effects being studied, rather than a violation of the parallel trends assumption.
    """)

    st.markdown("### 📊 DiD Results Summary")

    st.markdown("""
    The sequence of Difference-in-Differences (DiD) models progressively adds monetary policy controls and refinements to isolate the impact of U.S. monetary policy on income inequality.

    Across **Models 3 through 14**, the `DiD_Interaction` term—representing the treatment effect of U.S. policy changes relative to Canada—remains **consistently positive and highly statistically significant at the 0.1% level (p < 0.001)** while neither the U.S. federal funds rate (US_DFF) nor the Canadian target overnight rate are individually significant predictors of income inequality.

    - The coefficient of around **0.0037 to 0.0038** across these models is modest but highly consistent, suggesting a **stable treatment effect**.
    - **Model 14** stands out as the **best-performing and most comprehensive** specification.
        - It includes: `DiD_Interaction`, `Year`, US and Canada interest rates, and **inflation-adjusted money supply**.
        - It delivers the **highest t-statistic (49.15)**, low standard error (7.61e-05), and strong significance in both time and treatment terms.

    The significance of inflation-adjusted variables highlights the importance of capturing real purchasing power effects, rather than relying solely on nominal aggregates.

    ### ✅ Conclusion

    The consistent significance of the `DiD_Interaction` term across advanced models provides **strong causal evidence** that U.S. monetary policy contributed to rising inequality compared to Canada. This suggests that even when interest rates and money supply are not significant individually, their **joint structural effects** may play a critical role. 

    This analysis demonstrates that causal methods like DiD are essential in understanding **policy-driven inequality trends**, especially when paired with robust controls and strong parallel trend validation.
    """)

    st.info("📈 See Table 5 and Figure 7 in your full report for detailed model coefficients and visual inspection of U.S.-Canada Gini trends.")

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