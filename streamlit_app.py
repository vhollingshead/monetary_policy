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
from PIL import Image


# Set page config
st.set_page_config(page_title='Monetary Policy & Inequality', page_icon=':chart_with_upwards_trend:', layout='wide')


# Custom CSS for Styling
st.markdown(
    """
    <style>
    body { font-family: Arial, sans-serif; }
    .stButton>button { background-color: #4CAF50; color: white; border-radius: 8px; padding: 10px 20px; }
    .header { text-align: center; color: #4F7849; font-size: 24px; font-weight: bold; }
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

def image_display(image_path):
    image = Image.open(image_path)
    standard_width = 500
    aspect_ratio = image.height / image.width
    new_height = int(standard_width * aspect_ratio)
    resized_image = image.resize((standard_width, new_height))

    # Center image using columns
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(resized_image, use_container_width=True)

def display_caption(caption_text):
    st.markdown(
        f"<p style='text-align: center; font-size: 0.9rem; color: #555;'>{caption_text}</p>",
        unsafe_allow_html=True
    )

def home():
    # Header Section
    st.markdown("<div class='header'>Gini-Lab: An Interactive Dashboard on Monetary Policy’s Impact on Income Inequality</div>", unsafe_allow_html=True)
    st.markdown("<div class='subheader'>Nicole Kan, Victoria Hollingshead, William Lei, Tracy Volz</div>", unsafe_allow_html=True)

    # Problem Section
    st.markdown("### The Problem")
    problem_image_path = "images/theproblem_increase_gini.png"
    problem_caption_text = "Figure 1. Rising Inequality: The Gini Coefficient's Steady Climb from 1976 to 2023"
    image_display(problem_image_path)
    problem_caption = display_caption(problem_caption_text)

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
    We introduce the **The Gini Lab**, an interactive tool to highlight the impact of monetary policy on income inequality in the United States. It enables users to:

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
    The Gini Lab is an interactive platform that makes the link between monetary policy and income inequality accessible and actionable. It provides real-time insights into how policy decisions, such as interest rate changes, affect income inequality in the United States. 
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

    st.title("Meet Our Team")

    for item in images:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(item["url"], use_container_width=True)
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

    with st.expander("💻 Model Overview"):
        st.markdown("""

        - **Time Series Model for Forecasting:** We use a time series model to predict future trends in the Gini coefficient based on monetary policy changes. The model analyzes historical trends and allows users to simulate scenarios, such as adjusting interest rates, to see their impact on inequality over time. Results are visualized in the dashboard, showing both historical and forecasted trends.

        - **Machine Learning Model (LSTM) to Predict the Current Gini**: A Long Short-Term Memory (LSTM) model, a type of machine learning approach, predict the current Gini coefficient using recent economic data. This provides an up-to-date snapshot of income inequality, which is integrated into the dashboard for real-time insights. By combining forecasting with real-time analysis, the Monetary Policy Dashboard offers a comprehensive view of income inequality dynamics, empowering users to explore both current and future impacts of monetary policy.

        - **RandomForestRegressor to Predict the Current Gini (Future Effort)**: The RandomForestRegressor is an ensemble model composed of multiple trees, with each tree trained on a bootstrapped sample of the training data and random subset of the features. Final predictions are aggregated as a final step. While this model is not represented in the dashboard, it has shown considerable performance improvement over traditional time series and models. We recommend future teams explore supervised learning options like RandomForestRegressor for further analysis. 

        - **Difference-in-Differences (DiD) for Causal Inference**: To assess the causal impact of monetary policy on income inequality, we use a Difference-in-Differences (DiD) approach. This method compares changes in inequality metrics, like the Gini coefficient, between a treatment group (e.g., regions or groups affected by a specific policy change, such as an interest rate hike) and a control group (e.g., regions or groups not affected) before and after the policy intervention. By isolating the policy’s effect, DiD helps us understand how much of the change in inequality can be attributed to the monetary policy itself, rather than other factors.
        """)

    with st.expander("🛠️ Data Pipeline & Preprocessing"):
        st.markdown("""
        - **Data Collection:** We source economic indicators from the Federal Reserve Economic Data (FRED), including interest rates, consumer price indices, and asset levels. Financial market data, such as equity indices, is retrieved from Yahoo Finance. Inequality metrics, like the Gini coefficient, are obtained from public datasets (e.g., real-time inequality databases).
        - **Data Types**: We targeted economic variables that fall into the two major categories – Leading and Lagging.  Leading variables can be thought of as levers the federal reserve can pull to trigger a response.  They are often used to make forward-looking decisions.  Lagging variables can be thought of as measurements of economic performance and are often used to confirm economic shifts have occurred.  
        - **Preprocessing**: Multiple datasets are merged into a unified DataFrame, aligning them by date to create a comprehensive time series dataset for analysis.
        - **FRED API**: For the dashboard, data is pulled directly from the FRED database using the public API. Because leading and lagging variables have varying frequencies, they are averaged over the last month, published on the dashboard, and entered to the LSTM model for prediction.
        
        Data preprocessing prepares the raw data for modeling by ensuring consistency and quality. The high-level steps include:
                    
        - **Data Alignment:** Datasets are merged by date, ensuring all time series are synchronized. For example, FRED data (e.g., interest rates) is combined with inequality metrics and financial data.
        - **Frequency Adjustment:** Data is resampled to a consistent frequency (e.g., daily or monthly) to match the analysis requirements, using methods like forward-filling for missing values.
        - **Handling Missing Values:** Missing data points are identified and addressed, either by forward-filling or filtering out incomplete records, to ensure a complete dataset.
        - **Format Conversion:** Dates are converted to a standard datetime format, and the dataset is indexed by date for time series analysis.
        - **Data Split:** Our data was split chronologically (rather than randomly) to retain temporal relationships inherent in the data and to ensure past data was used to make sequential predictions in the future.  In addition, the temporal split establishes a “maximum forecast horizon” that serves as a guideline for model retraining (Hyndman, 2021).
        """)

        st.markdown("""
        

       
        """)


    with st.expander("📐 Gini Coefficient Calculation"):
        st.markdown("""
        The Gini coefficient is derived from the annual socioeconomic data produced by Blanchet et al. (2022), where income share for working-age adults in the **bottom 50%**, **middle 40%**, and **top 10%** are outlined on a quarterly basis between 1976 and 2023. Figure 2 shows the Gini coefficient formula applied. As a first step, we set the index (i) such that the socioeconomic bins were ascended by group (i.e., ordered from the bottom 50% to the middle 40% to the top 10%). We then computed the Gini coefficient while accounting for the population count for each socioeconomic bin. This essentially “weighted” our income share by accounting for the actual population distribution. It is important to note that this approach does not provide the most granular Gini coefficient, but is effective in producing a relative estimate based on limited information. 
        """)
        st.latex(r""" G = \frac{\sum_{i=1}^{n} (2i - n - 1) x_i}{n \sum_{i=1}^{n} x_i} """)
        gini_caption_text = "Figure 2. Gini coefficient formula in which: n = total population; i = ith individual, individuals are ranked from lowest to highest income ; x_i = income share of ith individual."
        display_caption(gini_caption_text)

    with st.expander("📈 Model Evaluation Summary"):
        st.markdown("We explored four key machine learning models over the course of this study to capture trends in inequality given direct and indirect economic indicators: SARIMAX, LSTM, XGBoost, and RandomForestRegressor. Below are the comparative performance of each with regard to MAE, MSE, and RMSE.")
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

        model_summary_table_caption = "Table 1: Performance Metrics of Models Trained on Direct and Indirect Economic Indicators"
        display_caption(model_summary_table_caption)

        st.markdown("""
        While the **RandomForestRegressor** outperformed traditional time series models, we selected **ARIMAX** for forecasting the future gini coefficient and **LSTM** to predict the current gini coefficient for their conventional use in industry and relative strong performance. During the training period of the forecasting model, we compared ARIMA and SARIMAX models for forecasting the Gini coefficient time series, both configured with the same time series order based on ACF and PACF plots. The baseline ARIMA model relied solely on the Gini coefficient data, while SARIMAX incorporated exogenous variables—the effective federal funds rate (DFF) and M2 money supply (M2)—to enhance its predictive power. Based on the performance metrics in Table1 below, ARIMAX outperformed ARIMA, demonstrating that the inclusion of DFF and M2 as exogenous variables significantly improved the model’s ability to capture the dynamics of the Gini coefficient.
        
        During the training period for the model to predict the current gini coefficient, we compared LSTM time series networks with traditional linear regression. Both models used the same feature set—mortgage-backed securities (WSHOMCB), Balance Sheet, Total Assets (QBPBSTASSCMRTSEC), and Consumer Credit, Student Loans (FGCCSAQ027S)—and identical data splits. While linear regression provided a straightforward baseline, the LSTM architecture significantly outperformed it according to metrics in Table 2. The LSTM's ability to capture temporal dependencies and non-linear relationships between economic indicators and inequality proved superior to linear regression's static approach. This demonstrates that the sequential processing capabilities of LSTM networks are particularly valuable for modeling the complex dynamics of income inequality as measured by the Gini coefficient.  We will describe more details of the model under the “Long Short Term Memory (LSTM)” section below. 
        
        Additional details of the Random Forest Regressor are described below under the section titled “Random Forest Regressor.”
        """)

    with st.expander("📉 Time Series Forecasting Model"):
        st.markdown("""
        The time series modeling process aims to forecast the Gini coefficient, a key measure of income inequality, and assess the impact of monetary policy. We begin with a simple ARIMA model due to its suitability for capturing temporal patterns in time series data like the Gini coefficient, which exhibits trends and autocorrelation. After evaluating ARIMA, we transition to ARIMAX to incorporate exogenous variables—specifically the logarithms of the Federal Funds Rate (log DFF) and U.S. M2 Money Supply (log M2)—as critical monetary indicators. Our goal is not only to predict the Gini coefficient but to understand how these policy variables influence income inequality. The process involves:
        
        - Testing for stationarity using the Augmented Dickey-Fuller (ADF) test, which revealed the Gini coefficient was non-stationary (p-value > 0.05), requiring first-differencing.
        - Analyzing ACF/PACF plots to determine the ARIMA order (1,1,1).
        - Fitting an ARIMA model as a baseline, then extending to SARIMAX to include log DFF and log M2.

        The ARIMA model serves as the baseline, evaluated using Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE) on the training set. During the training period, the baseline ARIMA(1,1,1) showed reasonable performance for a simple model, while SARIMAX, incorporating log DFF and log M2, outperformed it (as shown in the table), demonstrating that these exogenous variables better capture the Gini coefficient’s dynamics. The final SARIMAX model, using log DFF and log M2, was evaluated on a train-validation-test split (80-10-10), yielding improved predictive accuracy.
                    
        """)

        arimax_data = {
            "Dataset": ["Training Set", "Validation Set", "Test Set"],
            "MAE": [0.000661, 0.002192, 0.000485],
            "MSE": [0.000001, 0.000016, 0.000001],
            "RMSE": [0.000836, 0.004004, 0.000563]
        }

        evaluation_df = pd.DataFrame(arimax_data)
        
        arimax_image2 = "images/Figure 3_ARIMAX Forecast vs. Actual Differenced Gini Coefficient.png"

        arimaxcaption1 = "Table 2: ARIMA and ARIMAX Performance Metrics for Gini Coefficient Forecasting"
        arimaxcaption2 = "Figure 3: ARIMAX Forecast vs. Actual Differenced Gini Coefficient (2014-2023)"
        
        st.dataframe(evaluation_df, use_container_width=True)
        display_caption(arimaxcaption1)

        image_display(arimax_image2)
        display_caption(arimaxcaption2)


        st.markdown("""
        The ARIMAX model effectively captures the directional trends of the differenced Gini coefficient from 2014 to 2023, as seen in the alignment of forecasted and actual values during the validation and test periods. However, it struggles to handle the heightened volatility during the COVID-19 period around 2020-2022, where the actual test data exhibits sharp fluctuations that the model fails to predict accurately, resulting in a smoother forecast that underestimates the extreme variations.

        **Key Learnings & Impact**
                    
        The improved metrics suggest SARIMAX better captures the relationship between monetary policy and inequality, though our focus remains on explanatory power rather than predictive accuracy. As shown in the forecast plot, the ARIMAX model, using log DFF and log M2, is effective at predicting the direction of the differenced Gini coefficient, with the starting and ending points of the actual and forecasted values aligning closely in both validation and test periods. However, it struggles to capture the volatility during the COVID period, where extreme monetary policies led to significant fluctuations in the Gini coefficient.

        The modeling process revealed that monetary policy variables, specifically the federal funds rate and M2 money supply, have some level of influence in predicting the Gini coefficient, with ARIMAX providing a clearer understanding of their impact compared to ARIMA. Overall, we value the explanation of monetary policy tools’ impact on the Gini coefficient over achieving a perfect statistical fit. This aligns with our goal of assessing how policy decisions affect income inequality, rather than solely predicting the Gini coefficient. The dashboard’s ability to simulate these effects empowers policymakers to design more equitable monetary policies, potentially reducing economic disparity and fostering sustainable growth.

        """)

    with st.expander("🧠 LSTM Model to Predict the Current Gini"):
        st.markdown("""
        A comprehensive hyperparameter tuning study was conducted to optimize LSTM models for predicting the current Gini coefficient based on key economic indicators. The research focused on three significant FRED sourced features that showed strong correlations with income inequality:

        - WSHOMCB (Mortgage-Backed Securities)
        - QBPBSTASSCMRTSEC (Balance Sheet with Total Assets)
        - FGCCSAQ027S (Consumer Credit including Student Loans)

        These features were selected through a multi-stage feature selection process: first identifying variables with strong correlation coefficients with the Gini coefficient, then ranking feature importance using random forest models, and finally refining the selection using ridge regression to eliminate redundant or less impactful predictors.
                    
        The sequential nature of LSTM models proved particularly significant for this application, as economic indicators and inequality metrics inherently contain temporal patterns and dependencies. By processing data as sequences, the LSTM model could effectively capture how changes in mortgage-backed securities, balance sheet with total assets, and consumer credit including student loans over time influence income inequality trends, rather than treating each time point as an independent observation. This temporal awareness allows the model to detect lagged effects and cumulative impacts that traditional regression methods cannot adequately address.

        The study explored two distinct data splitting approaches: a traditional 80:10:10 split and an optimized 72:19:9 split determined through ridge regression analysis to minimize the variability of Mean Absolute Error (MAE) across test sets. This optimized split was specifically designed to address extreme variability observed in the validation dataset while maintaining tight test set variability. Importantly, data scaling was performed prior to splitting purely to reduce computational load during the extensive hyperparameter tuning process. While this approach could potentially introduce data leakage concerns, a comprehensive data leakage check was performed and successfully passed, confirming the integrity of the model evaluation process.

        The most successful architecture (Model #11) utilized a sequential structure with two LSTM layers (64 units each, ReLU activation functions) followed by dense layers with ELU and linear activations. This model incorporated dropout regularization (25%) and L2 regularization (0.001) to prevent overfitting, while employing the Huber loss function to manage outliers effectively. Each model was averaged over 5 runs to promote stability in the results, ensuring that performance metrics were not influenced by random initialization effects and increasing confidence in the consistency of the model's performance and the efficacy of the identified optimal hyperparameters.

        Performance metrics clearly demonstrated the superiority of LSTM models over traditional linear regression approaches. The best-performing LSTM configuration achieved a test Mean Absolute Error (MAE) of just 0.001133, dramatically outperforming the baseline linear regression model's 0.011027 MAE - a nearly 12-fold improvement in prediction accuracy.            

        This research highlights how deep learning techniques can significantly enhance our ability to forecast economic inequality metrics, potentially providing policymakers with more precise tools for evaluating the potential impact of economic policies on income distribution.
                           

        """)

        data = {
            "Dataset": ["Training Set", "Validation Set", "Test Set"],
            "MAE": [0.001774, 0.002289, 0.001133],
            "MSE": [0.000005, 0.000009, 0.000002],
            "RMSE": [0.002226, 0.003013, 0.001385]
        }

        lstm_eval_df = pd.DataFrame(data)

        st.dataframe(lstm_eval_df, use_container_width=True)
        lstm_caption1 = "Table 3: LSTM Model Performance Metrics Across Training, Validation, and Test Sets"
        display_caption(lstm_caption1)

        lstm_url = "images/Figure 4_Ensemble vs. Individual Models for Gini Coefficient Prediction.png"

        lstm_caption2 = "Figure 4: Ensemble vs. Individual Models for Gini Coefficient Prediction (2006-2024)"

        image_display(lstm_url)
        display_caption(lstm_caption2)




    with st.expander("🌲 RandomForestRegressor to Predict the Current Gini (Future Effort)"):
        st.markdown("""
        While the RandomForestRegressor may not be a dedicated time series model, this exploratory study found it was highly effective in predicting future Gini coefficients given time series data. Prior to training the RandomForestRegressor, the dataset was converted to a format suitable for supervised machine learning, in which lagged versions of each feature were created as new input variables. For these runs, we formulated a one-step-ahead forecast using the past 12 time steps. In other words, each feature was transformed to create 12 additional lagged variables (e.g, t-12, t-11, ..., t-1). After training the RandomForestRegressor with 1000 trees, we simulated a realistic forecasting scenario in which the model was retrained as soon as new data became available. For example, if testing data is denoted as [t1, t2, t3, t4], after a prediction was made and stored for t1, t1 was appended to the training data and used to retrain the model in preparation for the prediction for t2. 
                    
        The model's strong performance and ability to capture complex and nonlinear relationships makes it an available alternative to time-series specific models such as ARIMAX or LSTM. In Figure 5 below, abrupt dips in prediction are shown to occur, though absolute differences are less than 0.008 Gini coefficient points. In addition, the model was only retrained every 4 predictions; results may prove more reliable with more frequent retraining.
                    
        """)
        randomforest_image1 = "images/Figure 5_RandomForestRegressor Actual vs. Prediction on indirect indicators only.png .png"
        randomforest_image2 = "images/Figure 6_RandomForestRegressor Actual vs. Prediction results on direct indicators only, DFF and US M2 Supply.png"

        randomforestcaption1 = "Figure 5. RandomForestRegressor Actual vs. Prediction on indirect indicators only. "
        randomforestcaption2 = "Figure 6. RandomForestRegressor Actual vs. Prediction results on direct indicators only, DFF and US M2 Supply"
        
        image_display(randomforest_image1)
        display_caption(randomforestcaption1)

        image_display(randomforest_image2)
        display_caption(randomforestcaption2)

    with st.expander("📊 Difference-in-Differences (DiD) for Causal Inference"):
        st.markdown("""
        To assess the causal impact of monetary policy on income inequality, we use a **Difference-in-Differences (DiD)** approach. This method compares changes in inequality metrics, like the Gini coefficient, between a treatment group (e.g., regions or groups affected by a specific policy change, such as an interest rate hike) and a control group (e.g., regions or groups not affected) before and after the policy intervention. By isolating the policy’s effect, DiD helps us understand how much of the change in inequality can be attributed to the monetary policy itself, rather than other factors.
        """)

    with st.expander("🔮 Forecasting Framework (ARIMAX-Based)"):
        st.markdown("""
        The **forecasting framework** leverages our learned ARIMAX model to predict the Gini coefficient over a multiple-year horizon, starting from the last observed data point. The process involves:


        - **Model Setup**: An ARIMAX model is fitted on the full dataset, using the first-differenced Gini coefficient as the dependent variable and Federal Funds Rate (DFF) and U.S. M2 Money Supply (US_M2_USD) as exogenous variables to capture monetary policy effects.
        - **Future Variables**: Future values of DFF and M2 are estimated using simple assumptions: DFF is scaled based on its last value plus an annual change, while M2 grows at its historical annual rate.
        - **Forecast Output**: The model generates forecasts for the Gini coefficient, including point estimates and 95% confidence intervals.

        In our Monetary Policy Dashboard, the ARIMAX model allows users to forecast the Gini coefficient over the next five years by inputting custom monetary policy assumptions, with 95% confidence intervals shaded around the forecast. Users can specify the annual change in the Federal Funds Rate and the annual growth rate of M2, enabling flexible exploration of monetary policy impacts on income inequality.

        We have an example below that demonstrates this forecasting capability, using user-defined inputs where the Federal Funds Rate decreases by 1% annually (annual_dff_change=-1) and M2 grows at an annual rate of 10% (annual_m2_growth_pct=0.1). The historical Gini coefficient (blue line) reflects the trend from 2006 to 2023, showing rising inequality over time, while the forecasted Gini coefficient (red dashed line) projects a continued increase in inequality under these assumptions, with 95% confidence intervals shaded in red.

        To improve model stability and reliability, we applied a weighting adjustment to reduce the influence of the COVID period, which introduced extreme volatility and structural distortions in both macroeconomic variables and the Gini trend. Research has shown that models trained on highly irregular periods can overfit to outliers and fail to generalize well. By incorporating this weighting scheme, the model better reflects long-term macroeconomic relationships rather than transient pandemic-era shocks.
                    
        This example below highlights the dashboard’s ability to simulate user-defined monetary policy impacts on income inequality, though it is not intended to provide specific policy recommendations.

        """)
        forecast_image = "images/Figure 8_Example Dashboard Forecast of Gini Coefficient with 95% Confidence Interval.png"
        image_display(forecast_image)
        forecast_caption = "Figure 8: Example Dashboard Forecast of Gini Coefficient with 95% Confidence Interval"
        display_caption(forecast_caption)

    with st.expander("⚠️ Limitations"):
        st.markdown("""
        The time series forecasting component of the Monetary Policy Dashboard has the following limitations:
        - **Simplified Assumptions**: The ARIMAX model relies on historical data and basic assumptions for future exogenous variables like the Federal Funds Rate (DFF) and M2 Money Supply, which may not account for unexpected economic shocks or policy shifts.
        - **Heterogeneity:** The model aggregates data at a national level, potentially overlooking heterogeneity across different economic regions or demographic groups.
        - **Non-Normal Residuals:** The residuals are not entirely normal, which may affect the reliability of confidence intervals for forecasts.
        - **Stationarity Concerns:** The assumption of stationarity after differencing may not hold over long horizons, potentially impacting forecast accuracy.
        - **Illustrative Purpose:** While the dashboard effectively illustrates the potential impact of monetary policy on income inequality, its scenario-based forecasts should be interpreted as illustrative rather than definitive predictions.
        """)

    with st.expander("🔁 Model Retraining Strategy"):
        st.markdown("""
         Our ARIMAX and LSTM models require different retraining frequencies based on economic conditions:
        - **Normal Conditions**: Annual retraining is sufficient during stable periods. Since our “maximum forecast horizon” for the LSTM is ~1.7 years, we establish a conservative retraining schedule of once every 12 months for scenarios when economic factors remain consistent.
        - **Volatile Periods**: Quarterly retraining becomes necessary during volatility (like COVID-19), when economic relationships shift rapidly and model performance deteriorates within months rather than years.
        - **Trigger Conditions**: To monitor these risks, we propose non-scheduled model retraining when the fed funds rate changes more than 50 basis points in a quarter, and / or the economy experiences significant shifts in M2 money supply (ex: monthly growth rate greater than 1% for 3 or more months in a row).
        """)

    with st.expander("📌 Conclusion"):
        st.markdown("""
        This capstone demonstrates that monetary policy not only shapes macroeconomic conditions but also plays a measurable role in income inequality. Using ARIMAX models with exogenous variables like the Federal Funds Rate and M2 money supply, we significantly improved Gini coefficient forecasting accuracy. However, these models struggled to capture volatility during periods of economic disruption, such as the COVID-19 shock, underscoring the limits of linear forecasting frameworks under structural breaks.


        For real-time estimation, the LSTM model proved highly effective and achieved a huge improvement over linear regression. Its capacity to model non-linear temporal dependencies between financial indicators and inequality metrics reinforces the value of deep learning in economic forecasting. RandomForestRegressor shows promising performance and opens up the possibility of including supervised learning techniques in effectively capturing temporal dependencies. 


        Meanwhile, the Difference-in-Differences (DiD) analysis provided robust causal evidence that U.S. monetary policy contributed to rising inequality, especially relative to Canada. Despite this, traditional monetary levers like interest rates and M2 did not individually emerge as significant predictors in the causal framework, suggesting broader structural dynamics are at play.


        Together, these results highlight that advanced statistical and machine learning models can provide deep insights into complex policy effects. Yet, they also emphasize critical challenges—such as model fragility during volatile periods, difficulty in isolating causality, and the need for continual retraining in dynamic economic contexts. Moving forward, integrating ensemble learning (e.g., Random Forest), modeling real-time shocks, and testing alternate causal frameworks like Regression Discontinuity can further enhance the explanatory and predictive power of policy tools aimed at economic equity

        """)

    with st.expander("📚 References"):
        st.markdown("""
        Blanchet, T., Saez, E., & Zucman, G. (November 2022). *Who benefits from income and wealth growth in the United States?* Department of Economics, University of California, Berkeley.  
        
        Hyndman, R. J., & Athanasopoulos, G. (2021). *5.8 Training and test sets*. In Forecasting: Principles and practice. OTexts.
        """)



def causal_inf():
    st.markdown("## Causal Inference")

    st.markdown("""
    To examine the impact of U.S. monetary policy on income inequality, the analysis employs a series of Difference-in-Differences (DiD) models using Canada as a control group. The core idea is to compare changes in the U.S. Gini coefficient over time—before and after key monetary policy events—with corresponding changes in Canada, a country with a similar economic structure but independent monetary policy. By focusing on the interaction between treatment status (U.S. vs. Canada) and time, these models aim to isolate the causal effect of U.S. monetary shifts (such as changes in interest rates or money supply) on income inequality trends.
    
    A baseline DiD model includes just the treatment dummy, time trend, and an interaction term (**DiD_Interaction**), but more refined models introduce monetary policy variables like the U.S. federal funds rate (US_DFF), the interest rate spread between the U.S. and Canada, and both nominal and inflation-adjusted money supply (M2). As models evolve in complexity—e.g., Model 14 includes interest rate changes, cross-country spreads, and inflation-adjusted M2—the results become more robust and the DiD_Interaction term gains strong statistical significance. This suggests that U.S. monetary policy may indeed play a role in shaping inequality trends.
    
    To assess the validity of the parallel trends assumption—a key requirement for causal inference in Difference-in-Differences (DiD) analysis—a dynamic event study model was employed. A traditional pre-trend check was not feasible due to missing Gini coefficient data for Canada prior to 1992. Instead, the event study specification evaluates whether the U.S. and Canada followed similar inequality trends leading up to major monetary policy events, particularly around 2008. The interaction term **Treatment:Event_Time** captures the differential trend between the U.S. and Canada over time. In the model, this coefficient is small (0.0245) and statistically insignificant (p = 0.249), indicating no strong evidence of divergence in trends prior to the treatment period.
    
    This supports the parallel trends assumption and reinforces the credibility of Canada as a valid control group. Additionally, the high R-squared (0.977) suggests that the model fits the data well, and the insignificant Event_Time term (p = 0.102) further suggests that overall trends were not drastically shifting in the pre-treatment years. Visual inspection of the Gini trends further supports this conclusion, as the U.S. and Canadian series appear to move in parallel throughout the pre-2008 period with no major divergence. While some divergence emerges around 2015—with U.S. inequality trending upward and Canada’s trend flattening or declining—this occurs well into the post-treatment period. Since the models estimate treatment effects over the full 1992–2019 window, such divergence may in fact reflect the very effects being studied, rather than a violation of the parallel trends assumption. Together, the statistical and graphical evidence provide strong justification for using Canada as a control group in the DiD framework.
    """)

    st.markdown("### 📊 Parallel Trend Assumption Results")
    did_image = "images/Table 4_OLS Regression Results for Parallel Trend Assumption in Gini Coefficient Analysis.png"
    image_display(did_image)
    did_caption_title = "Table 4: OLS Regression Results for Parallel Trend Assumption in Gini Coefficient Analysis"
    display_caption(did_caption_title)

    did_image_fig7 = "images/Figure 7_Parallel Trends Check for US vs. Canada Gini Coefficients.png"
    image_display(did_image_fig7)
    did_caption_title_fig7 = "Figure 7: Parallel Trends Check for US vs. Canada Gini Coefficients (1995-2023)"
    display_caption(did_caption_title_fig7)

    st.markdown("### 📊 DiD Results Summary")

    # Create a DataFrame for DiD Results Summary
    data = {
        "Model": [f"Model {i}" for i in range(1, 15)],
        "Variables": [
            "Treatment + Year + DiD_Interaction + US_DFF",
            "Treatment + Year + DiD_Interaction + US_DFF + Canada_Overnight_Target_Rate",
            "DiD_Interaction + Year + US_DFF + Canada_Overnight_Target_Rate + log_US_M2_USD + log_Canada_M2_USD",
            "DiD_Interaction + Year + US_DFF + Canada_Overnight_Target_Rate + log_US_M2_USD",
            "DiD_Interaction + Year + US_DFF + log_US_M2_USD",
            "Year + DiD_Interaction + US_DFF_change",
            "DiD_Interaction + Year + US_DFF_change + Canada_Rate_change",
            "DiD_Interaction + Year + Interest_Spread",
            "Year + DiD_Interaction + US_DFF_change + log_US_M2_USD",
            "DiD_Interaction + Year + US_DFF_change + Canada_Rate_change + log_US_M2_USD",
            "DiD_Interaction + Year + Interest_Spread + log_US_M2_USD",
            "DiD_Interaction + Year + US_DFF_change + Canada_Rate_change + Interest_Spread + log_US_M2_USD",
            "DiD_Interaction + Year + US_DFF_change + Canada_Rate_change + log_US_M2_USD_inflation_adj",
            "DiD_Interaction + Year + US_DFF + Canada_Overnight_Target_Rate + log_US_M2_USD_inflation_adj + log_Canada_M2_USD"
        ],
        "Coef": [0.0274, 0.0274, 0.0037, 0.0037, 0.0037, 0.0038, 0.0038, 0.0038, 0.0038, 0.0038, 0.0038, 0.0038, 0.0038, 0.0037],
        "Std Err": [0.021, 0.02, 8.12e-05, 8.16e-05, 8.38e-05, 8.15e-05, 8.15e-05, 8.20e-05, 8.23e-05, 8.21e-05, 8.23e-05, 8.10e-05, 7.98e-05, 7.61e-05],
        "t": [1.276, 1.37, 46.075, 45.853, 44.651, 46.025, 46.064, 45.767, 45.563, 45.709, 45.579, 46.303, 47.037, 49.15],
        "P>|t|": [0.208, 0.177, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "Significant Term(s)": [
            "DiD_Interaction, Year *",
            "DiD_Interaction, US_DFF *, Canada_Overnight_Target_Rate **",
            "DiD_Interaction ***",
            "DiD_Interaction ***",
            "DiD_Interaction ***, Year *, Intercept *, log_US_M2_USD *",
            "DiD_Interaction ***, Intercept *, Year ***",
            "DiD_Interaction ***, Intercept *, Year ***",
            "DiD_Interaction ***, Year **",
            "DiD_Interaction ***",
            "DiD_Interaction ***",
            "DiD_Interaction ***",
            "DiD_Interaction ***",
            "DiD_Interaction ***",
            "DiD_Interaction ***, Year **"
        ]
    }

    did_results_df = pd.DataFrame(data)



    st.dataframe(did_results_df, use_container_width=True)

    did_tabl5 = "Table 5: Difference-in-Differences (DiD) Results Summary for Gini Coefficient Analysis"
    display_caption(did_tabl5)

    st.markdown("""
    **Legend:**
                
    3 asterisks = p < 0.001  
    2 asterisks = p < 0.01  
    1 asterisk = p < 0.05  
    ns = not significant
    
    
    """)

    st.markdown("""
    The sequence of Difference-in-Differences (DiD) models progressively adds monetary policy controls and refinements to isolate the impact of U.S. monetary policy on income inequality. Across Models 3 through 14, the DiD_Interaction term—representing the treatment effect of U.S. policy changes relative to Canada—remains consistently positive and highly statistically significant at the 0.1% level (p < 0.001) while neither the U.S. federal funds rate (US_DFF) nor the Canadian target overnight rate are individually significant predictors of income inequality. This suggests robust evidence that inequality in the U.S. increased more than in Canada following monetary policy shifts. The coefficient of around 0.0037 to 0.0038 across these models, while modest in magnitude, is precise and consistent, pointing to a stable treatment effect across various model specifications and control variables.

    Among the models, Model 14 stands out as the best-performing and most comprehensive specification. It includes not only the DiD interaction and time trend (Year), but also U.S. and Canadian interest rates (US_DFF, Canada_Overnight_Target_Rate), log of inflation-adjusted US money supply and log of Canada money supply. This model yields the highest t-statistic (49.15) for the DiD term, a low standard error (7.61e-05), and significance in the Year variable as well (p < 0.01), indicating that both the time dimension and treatment effect are driving variation in inequality. Moreover, this model includes real (inflation-adjusted) monetary aggregates, which add important explanatory power by capturing the purchasing power dynamics of monetary policy—something nominal M2 measures miss.
                
    The consistent significance of the DiD_Interaction term across all advanced models (Models 3–14) provides strong evidence that U.S. monetary policy had an effect on income inequality, compared to Canada, a similar developed economy not directly affected by Federal Reserve policy. On the other hand, the Federal Funds Rate and M2 were not significant drivers, suggesting that broader economic factors contributed to inequality changes, rather than just monetary policy alone.

    """)

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