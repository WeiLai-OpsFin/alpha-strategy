#!/usr/bin/env python3
"""
FIN3083 Group Project - Data Download Script
Strategy: Macro-Liquidity Driven Regime-Switching Factor Rotation

This script downloads all required data from Tushare Pro API:
1. Shibor rates (macro liquidity indicator)
2. CSI 300 index constituents (historical weights)
3. Daily valuation metrics (PE, PB, dividend yield)
4. Financial indicators (ROE, growth rates)
5. Stock prices (open, high, low, close)
6. Index prices (CSI 300 benchmark)

Author: Group FIN3083
Date: 2026-04-17
"""

import tushare as ts
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import time
import warnings
warnings.filterwarnings('ignore')

# ==================== Configuration ====================
TUSHARE_TOKEN = '40ceb9d9603bf5dd49494eeb8b2acc5bd2b531306a4daae624e4eff2'
START_DATE = '20060101'
END_DATE = '20241231'
DATA_DIR = 'data'

# Create data directory
os.makedirs(DATA_DIR, exist_ok=True)

# Initialize Tushare
print("Initializing Tushare Pro...")
ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()

# Test connection
try:
    df_test = pro.trade_cal(exchange='SSE', start_date='20240101', end_date='20240110', limit=1)
    print("✓ Tushare connection successful")
except Exception as e:
    print(f"✗ Tushare connection failed: {e}")
    exit(1)

print("\n" + "="*60)
print("STARTING DATA DOWNLOAD")
print("="*60)

# ==================== Function 1: Download Shibor Rates ====================
def download_shibor():
    """Download Shibor rates for macro liquidity indicator"""
    print("\n[1/6] Downloading Shibor rates...")

    try:
        # Get Shibor data (3-month rate as main indicator)
        # Note: Shibor data starts from October 2006, earlier dates will be ignored
        df_shibor = pro.shibor(start_date=START_DATE, end_date=END_DATE)

        if df_shibor.empty:
            print("  ✗ No Shibor data retrieved")
            return None

        # Select relevant columns
        df_shibor = df_shibor[['date', '3m']].copy()
        df_shibor.rename(columns={'3m': 'shibor_3m'}, inplace=True)
        df_shibor['date'] = pd.to_datetime(df_shibor['date'])

        # Calculate 20-day moving average
        df_shibor = df_shibor.sort_values('date')
        df_shibor['shibor_ma20'] = df_shibor['shibor_3m'].rolling(window=20, min_periods=10).mean()

        # Save
        output_file = os.path.join(DATA_DIR, 'macro_shibor.csv')
        df_shibor.to_csv(output_file, index=False)

        print(f"  ✓ Shibor data: {len(df_shibor)} records")
        print(f"  ✓ Saved to: {output_file}")

        return df_shibor

    except Exception as e:
        print(f"  ✗ Error downloading Shibor: {e}")
        return None

# ==================== Function 2: Download CSI 300 Constituents ====================
def download_index_weights():
    """Download CSI 300 historical constituents to avoid survivorship bias"""
    print("\n[2/6] Downloading CSI 300 constituents...")

    try:
        # Get trading calendar
        df_cal = pro.trade_cal(exchange='SSE', start_date=START_DATE, end_date=END_DATE, is_open='1')
        trade_dates = df_cal['cal_date'].tolist()

        # Sample monthly (first trading day of each month)
        monthly_dates = []
        current_month = None
        for date in trade_dates:
            month = date[:6]
            if month != current_month:
                monthly_dates.append(date)
                current_month = month

        print(f"  → Sampling {len(monthly_dates)} monthly dates")

        all_weights = []
        for i, date in enumerate(monthly_dates):
            if i % 6 == 0:
                print(f"  Progress: {i}/{len(monthly_dates)} months...")

            try:
                df_weight = pro.index_weight(index_code='000300.SH', trade_date=date)
                if not df_weight.empty:
                    all_weights.append(df_weight)
                time.sleep(0.2)  # Rate limit
            except Exception as e:
                print(f"  Warning: Failed to get weights for {date}")
                continue

        if all_weights:
            df_weights = pd.concat(all_weights, ignore_index=True)
            output_file = os.path.join(DATA_DIR, 'index_weights.csv')
            df_weights.to_csv(output_file, index=False)

            print(f"  ✓ Constituents data: {len(df_weights)} records")
            print(f"  ✓ Covering {df_weights['trade_date'].nunique()} dates")
            print(f"  ✓ Saved to: {output_file}")

            return df_weights
        else:
            print("  ✗ No constituent data retrieved")
            return None

    except Exception as e:
        print(f"  ✗ Error downloading constituents: {e}")
        return None

# ==================== Function 3: Download Daily Valuation Metrics ====================
def download_daily_basic():
    """Download daily valuation metrics (PE, PB, dividend yield)"""
    print("\n[3/6] Downloading daily valuation metrics...")

    try:
        # Get CSI 300 constituents for filtering
        df_constituents = pd.read_csv(os.path.join(DATA_DIR, 'index_weights.csv'))
        unique_stocks = df_constituents['con_code'].unique().tolist()

        print(f"  → Downloading data for {len(unique_stocks)} stocks")

        all_basic = []
        batch_size = 50  # Process in batches

        for i in range(0, len(unique_stocks), batch_size):
            batch = unique_stocks[i:i+batch_size]
            print(f"  Progress: {i}/{len(unique_stocks)} stocks...")

            for ts_code in batch:
                try:
                    df_basic = pro.daily_basic(ts_code=ts_code, start_date=START_DATE, end_date=END_DATE,
                                              fields='ts_code,trade_date,pe,pb,dv_ratio,turnover_rate')
                    if not df_basic.empty:
                        all_basic.append(df_basic)
                    time.sleep(0.15)
                except Exception as e:
                    continue

        if all_basic:
            df_basic_all = pd.concat(all_basic, ignore_index=True)

            # Calculate EP (Earnings-to-Price) = 1/PE
            df_basic_all['ep'] = 1 / df_basic_all['pe']
            # Calculate BP (Book-to-Price) = 1/PB
            df_basic_all['bp'] = 1 / df_basic_all['pb']

            output_file = os.path.join(DATA_DIR, 'daily_basic.csv')
            df_basic_all.to_csv(output_file, index=False)

            print(f"  ✓ Valuation data: {len(df_basic_all)} records")
            print(f"  ✓ Saved to: {output_file}")

            return df_basic_all
        else:
            print("  ✗ No valuation data retrieved")
            return None

    except Exception as e:
        print(f"  ✗ Error downloading valuation data: {e}")
        return None

# ==================== Function 4: Download Financial Indicators ====================
def download_fina_indicator():
    """Download financial indicators (ROE, growth rates)"""
    print("\n[4/6] Downloading financial indicators...")

    try:
        # Get CSI 300 constituents
        df_constituents = pd.read_csv(os.path.join(DATA_DIR, 'index_weights.csv'))
        unique_stocks = df_constituents['con_code'].unique().tolist()

        print(f"  → Downloading financial data for {len(unique_stocks)} stocks")

        all_fina = []
        batch_size = 30

        for i in range(0, len(unique_stocks), batch_size):
            batch = unique_stocks[i:i+batch_size]
            if i % 100 == 0:
                print(f"  Progress: {i}/{len(unique_stocks)} stocks...")

            for ts_code in batch:
                try:
                    df_fina = pro.fina_indicator(ts_code=ts_code, start_date='20190101', end_date=END_DATE)
                    if not df_fina.empty:
                        # Select key indicators
                        df_fina = df_fina[['ts_code', 'ann_date', 'end_date', 'roe', 'grossprofit_margin',
                                          'dt_netprofit_yoy', 'dt_eps']].copy()
                        all_fina.append(df_fina)
                    time.sleep(0.2)
                except Exception as e:
                    continue

        if all_fina:
            df_fina_all = pd.concat(all_fina, ignore_index=True)
            output_file = os.path.join(DATA_DIR, 'fina_indicator.csv')
            df_fina_all.to_csv(output_file, index=False)

            print(f"  ✓ Financial data: {len(df_fina_all)} records")
            print(f"  ✓ Saved to: {output_file}")

            return df_fina_all
        else:
            print("  ✗ No financial data retrieved")
            return None

    except Exception as e:
        print(f"  ✗ Error downloading financial data: {e}")
        return None

# ==================== Function 5: Download Stock Prices ====================
def download_stock_prices():
    """Download daily stock prices"""
    print("\n[5/6] Downloading stock prices...")

    try:
        # Get CSI 300 constituents
        df_constituents = pd.read_csv(os.path.join(DATA_DIR, 'index_weights.csv'))
        unique_stocks = df_constituents['con_code'].unique().tolist()

        print(f"  → Downloading prices for {len(unique_stocks)} stocks")

        all_prices = []
        batch_size = 50

        for i in range(0, len(unique_stocks), batch_size):
            batch = unique_stocks[i:i+batch_size]
            print(f"  Progress: {i}/{len(unique_stocks)} stocks...")

            for ts_code in batch:
                try:
                    df_price = pro.daily(ts_code=ts_code, start_date=START_DATE, end_date=END_DATE)
                    if not df_price.empty:
                        all_prices.append(df_price)
                    time.sleep(0.15)
                except Exception as e:
                    continue

        if all_prices:
            df_prices_all = pd.concat(all_prices, ignore_index=True)

            # Calculate daily returns
            df_prices_all = df_prices_all.sort_values(['ts_code', 'trade_date'])
            df_prices_all['pct_chg'] = df_prices_all.groupby('ts_code')['close'].pct_change() * 100

            output_file = os.path.join(DATA_DIR, 'stock_prices.csv')
            df_prices_all.to_csv(output_file, index=False)

            print(f"  ✓ Price data: {len(df_prices_all)} records")
            print(f"  ✓ Saved to: {output_file}")

            return df_prices_all
        else:
            print("  ✗ No price data retrieved")
            return None

    except Exception as e:
        print(f"  ✗ Error downloading price data: {e}")
        return None

# ==================== Function 6: Download Index Prices ====================
def download_index_prices():
    """Download CSI 300 index prices as benchmark"""
    print("\n[6/6] Downloading index prices...")

    try:
        # CSI 300 Index
        df_index = pro.index_daily(ts_code='000300.SH', start_date=START_DATE, end_date=END_DATE)

        if df_index.empty:
            print("  ✗ No index data retrieved")
            return None

        # Calculate index returns
        df_index = df_index.sort_values('trade_date')
        df_index['index_return'] = df_index['close'].pct_change()

        output_file = os.path.join(DATA_DIR, 'index_prices.csv')
        df_index.to_csv(output_file, index=False)

        print(f"  ✓ Index data: {len(df_index)} records")
        print(f"  ✓ Saved to: {output_file}")

        return df_index

    except Exception as e:
        print(f"  ✗ Error downloading index data: {e}")
        return None

# ==================== Main Execution ====================
if __name__ == "__main__":
    print("\nData Download for FIN3083 Project")
    print("Strategy: Macro-Liquidity Driven Regime-Switching Factor Rotation")
    print(f"Period: {START_DATE} to {END_DATE}")
    print("Note: Shibor data starts from October 2006")

    # Download all data
    results = {
        'Shibor': download_shibor(),
        'Index Weights': download_index_weights(),
        'Daily Basic': download_daily_basic(),
        'Financial Indicators': download_fina_indicator(),
        'Stock Prices': download_stock_prices(),
        'Index Prices': download_index_prices()
    }

    # Summary
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)

    for name, df in results.items():
        status = "✓" if df is not None else "✗"
        count = len(df) if df is not None else 0
        print(f"{status} {name}: {count} records")

    print(f"\nAll data files saved in: {os.path.abspath(DATA_DIR)}/")
    print("\nNext step: Run Python code to perform backtesting")
    print("="*60)
