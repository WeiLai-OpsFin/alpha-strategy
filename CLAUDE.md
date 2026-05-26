# CLAUDE.md

This repository implements a FIN3083 quantitative strategy project: a macro liquidity-driven state-switching factor rotation strategy for the Chinese A-share market.

## Run commands

```bash
pip install -r requirements.txt
python src/FIN3083_Strategy_Final.py
```

Alternative entry point:

```bash
python src/main.py
```

## Core methodology

- Backtest period: 2015-2024
- In-sample: 2015-2019
- Out-of-sample: 2020-2024
- Universe: 120 CSI 300 stocks, sampled as 48 large + 42 mid + 30 small
- Monthly portfolio: equal-weighted top 25 stocks
- Regime indicator: 3-month Shibor, using the in-sample median threshold
- Factors: Growth, Value, Quality, LowVol, Dividend, Momentum
- Factor weighting: 2-year rolling Spearman IC/IR, by regime, with threshold/filter/cap safeguards
- Risk overlays: Risk-Off defensive allocation, volatility targeting, and loss protection

## Important files

- `src/python_codes/config.py`: all strategy parameters
- `src/python_codes/data_import.py`: reads six raw CSV files
- `src/python_codes/data_clean.py`: date parsing, return calculation, stratified universe selection
- `src/python_codes/macro_regime.py`: Shibor-based Risk-On / Risk-Off classification
- `src/python_codes/factor_calc.py`: factor construction and z-score standardization
- `src/python_codes/signal_gen.py`: Spearman IC, dynamic weights, monthly top-25 selection
- `src/python_codes/backtest.py`: backtest and three risk overlays
- `src/python_codes/metrics.py`: performance metrics
- `src/python_codes/visualization.py`: charts
- `src/python_codes/pipeline.py`: end-to-end workflow

## Data files expected locally

The raw CSV files are intentionally ignored by git.

```text
data/macro_shibor.csv
data/index_weights.csv
data/stock_prices.csv
data/daily_basic.csv
data/fina_indicator.csv
data/index_prices.csv
```
