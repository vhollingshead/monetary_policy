from fredapi import Fred
import streamlit as st
from pandas_datareader import data as pdr
import pandas as pd
from datetime import datetime, timedelta

####################

def fred_api_funct():
    fred_key = st.secrets["api_keys"]["my_api_key"]
    fred = Fred(fred_key)
    # Get today's date
    today = datetime.today()

    end_date = today.strftime('%Y-%m-%d')
    start_date = (today - timedelta(days=90)).strftime('%Y-%m-%d')

    # FRED series: Nominal Broad U.S. Dollar Index
    # https://fred.stlouisfed.org/series/DTWEXBGS
    dtwexbgs = pdr.DataReader("DTWEXBGS", "fred", start=start_date, end=end_date)
    dtwexbgs.columns = ['DTWEXBGS']

    # FRED series: Consumer Price Index for All Urban Consumers: All Items Less Food and Energy in U.S. City Average
    # https://fred.stlouisfed.org/series/CPILFESL
    cpilfesl = pdr.DataReader("CPILFESL", "fred", start=start_date, end=end_date)
    cpilfesl.columns = ['CPILFESL']

    # FRED series: Consumer Price Index for All Urban Consumers: Purchasing Power of the Consumer Dollar in U.S. City Average
    # https://fred.stlouisfed.org/series/CUUR0000SA0R
    cuur0000sa0r = pdr.DataReader("CUUR0000SA0R", "fred", start=start_date, end=end_date)
    cuur0000sa0r.columns = ['CUUR0000SA0R']

    # FRED series: Consumer Price Index for All Urban Consumers: Energy in U.S. City Average
    # https://fred.stlouisfed.org/series/CPIENGSL
    cpiengsl = pdr.DataReader("CPIENGSL", "fred", start=start_date, end=end_date)
    cpiengsl.columns = ['CPIENGSL']

    # FRED series: Consumer Price Index for All Urban Consumers: All Items in U.S. City Average
    # https://fred.stlouisfed.org/series/CPIAUCSL
    cpiaucsl = pdr.DataReader("CPIAUCSL", "fred", start=start_date, end=end_date)
    cpiaucsl.columns = ['CPIAUCSL']

    # FRED series: Federal Funds Effective Rate (DFF)
    # https://fred.stlouisfed.org/series/DFF
    fred_def = pdr.DataReader("DFF", "fred", start=start_date, end=end_date)
    fred_def.columns = ['DFF']

    # FRED series: Effective Federal Funds Rate (EFFR)
    # https://fred.stlouisfed.org/series/EFFR
    effr = pdr.DataReader("EFFR", "fred", start=start_date, end=end_date)
    effr.columns = ['EFFR']

    # FRED series: Interest Rates and Price Indexes; Effective Federal Funds Rate (Percent), Level (BOGZ1FL072052006Q)
    # https://fred.stlouisfed.org/series/BOGZ1FL072052006Q
    bogz1fl072052006q = pdr.DataReader("BOGZ1FL072052006Q", "fred", start=start_date, end=end_date)
    bogz1fl072052006q.columns = ['BOGZ1FL072052006Q']

    # Assets: Total Assets: Total Assets (Less Eliminations from Consolidation): Wednesday Level (WALCL)
    # https://fred.stlouisfed.org/series/WALCL
    walcl = pdr.DataReader("WALCL", "fred", start=start_date, end=end_date)
    walcl.columns = ['WALCL']

    # Assets: Securities Held Outright: Mortgage-Backed Securities: Wednesday Level (WSHOMCB)
    # https://fred.stlouisfed.org/series/WSHOMCB
    wshomcb = pdr.DataReader("WSHOMCB", "fred", start=start_date, end=end_date)
    wshomcb.columns = ['WSHOMCB']

    # Assets: Securities Held Outright: U.S. Treasury Securities: All: Wednesday Level (TREAST)
    # https://fred.stlouisfed.org/series/TREAST
    treast = pdr.DataReader("TREAST", "fred", start=start_date, end=end_date)
    treast.columns = ['TREAST']

    # FRED series: Real Estate Loans: Residential Real Estate Loans: Revolving Home Equity Loans, All Commercial Banks (RHEACBW027SBOG)
    # https://fred.stlouisfed.org/series/RHEACBW027SBOG
    rheacbw027sbog = pdr.DataReader("RHEACBW027SBOG", "fred", start=start_date, end=end_date)
    rheacbw027sbog.columns = ['RHEACBW027SBOG']

    # FRED series: Real Estate Loans, All Commercial Banks (REALLN)
    # https://fred.stlouisfed.org/series/REALLN
    realln = pdr.DataReader("REALLN", "fred", start=start_date, end=end_date)
    realln.columns = ['REALLN']

    # FRED series: Households; Net Worth, Level (BOGZ1FL192090005Q)
    # https://fred.stlouisfed.org/series/BOGZ1FL192090005Q
    bogz1fl1920900005q = pdr.DataReader("BOGZ1FL192090005Q", "fred", start=start_date, end=end_date)
    bogz1fl1920900005q.columns = ['BOGZ1FL192090005Q']

    # FRED series: Labor Force Participation Rate (CIVPART)
    # https://fred.stlouisfed.org/series/CIVPART
    civpart = pdr.DataReader("CIVPART", "fred", start=start_date, end=end_date)
    civpart.columns = ['CIVPART']

    # FRED series: Labor Force Participation Rate (CIVPART)
    # https://fred.stlouisfed.org/series/M2REAL
    m2real = pdr.DataReader("M2REAL", "fred", start=start_date, end=end_date)
    m2real.columns = ['M2REAL']

    # Merging the DataFrames
    fred_data = pd.merge(cpilfesl, dtwexbgs, left_index=True, right_index=True, how='left')
    fred_data = fred_data.merge(cuur0000sa0r, left_index=True, right_index=True, how='left')
    fred_data = fred_data.merge(cpiengsl, left_index=True, right_index=True, how='left')
    fred_data = fred_data.merge(cpiaucsl, left_index=True, right_index=True, how='left')
    fred_data = fred_data.merge(fred_def, left_index=True, right_index=True, how='left')
    fred_data = fred_data.merge(effr, left_index=True, right_index=True, how='left')
    fred_data = fred_data.merge(bogz1fl072052006q, left_index=True, right_index=True, how='left')
    fred_data = fred_data.merge(walcl, left_index=True, right_index=True, how='left')
    fred_data = fred_data.merge(wshomcb, left_index=True, right_index=True, how='left')
    fred_data = fred_data.merge(treast, left_index=True, right_index=True, how='left')
    fred_data = fred_data.merge(rheacbw027sbog, left_index=True, right_index=True, how='left')
    fred_data = fred_data.merge(realln, left_index=True, right_index=True, how='left')
    fred_data = fred_data.merge(bogz1fl1920900005q, left_index=True, right_index=True, how='left')
    fred_data = fred_data.merge(civpart, left_index=True, right_index=True, how='left')
    fred_data = fred_data.merge(m2real, left_index=True, right_index=True, how='left')


    print("this is fred_data the head")
    print(fred_data.head())

    # Count the number of NaN values in the DataFrame
    nan_count = fred_data.isna().sum().sum()
    print("Number of NaN values:", nan_count)

    # Convert index to period and then to daily frequency
    fred_data.index = pd.to_datetime(fred_data.index)
    fred_data = fred_data.resample('D').asfreq()

    # Forward-fill missing values
    fred_data_daily = fred_data.ffill()

    # Change the name of the index column to 'Date'
    fred_data_daily.index.name = 'Date'

    print("this is fred_data_daily the head")
    print(fred_data_daily.head())

    # monthly_start_avg = fred_data_daily.resample('M').mean()

    # return monthly_start_avg
    return fred_data_daily

