# Macro Liquidity-Driven State-Switching Factor Rotation Strategy

Chinese A-share factor rotation strategy.

This repository is organized as a reproducible strategy research project: clean source code, reproducible configuration, paper/slides PDFs, and generated output examples.

## What the strategy does

The strategy uses 3-month Shibor to split the market into two liquidity states:

| State | Rule | Portfolio idea |
|---|---|---|
| Risk-On | `Shibor_3M < in-sample median` | Normal equity allocation |
| Risk-Off | `Shibor_3M >= in-sample median` | 50% equity + 50% Shibor cash |

It then builds six daily factors:

| Factor | Definition |
|---|---|
| Growth | net profit year-over-year growth |
| Value | earnings-to-price + book-to-price |
| Quality | return on equity |
| LowVol | inverse 20-day volatility |
| Dividend | dividend yield |
| Momentum | 12-1 month momentum |

Factor weights are dynamic. They are estimated with a 2-year rolling Spearman IC/IR method, with an IC threshold, momentum filter, negative-IC filter, and 40% cap.

## Main settings

| Item | Value |
|---|---:|
| Backtest period | 2015-2024 |
| In-sample period | 2015-2019 |
| Out-of-sample period | 2020-2024 |
| Universe | 120 CSI 300 stocks |
| Stratified sample | 48 large + 42 mid + 30 small |
| Monthly selection | Top 25 stocks |
| Rebalancing | Last trading day of each month |
| IC window | 2 years |
| Volatility target | 20% annualized |
| Transaction cost | commission + stamp duty + impact |

## Folder structure

```text
.
├── README.md
├── requirements.txt
├── src/
│   ├── FIN3083_Strategy_Final.py
│   ├── main.py
│   └── python_codes/
│       ├── config.py
│       ├── data_import.py
│       ├── data_clean.py
│       ├── macro_regime.py
│       ├── factor_calc.py
│       ├── signal_gen.py
│       ├── backtest.py
│       ├── metrics.py
│       ├── visualization.py
│       ├── pipeline.py
│       └── sensitivity.py
├── data/
│   └── README.md
├── output/
│   ├── performance_summary.csv
│   ├── regime_sharpe.csv
│   ├── ic_stability.csv
│   ├── daily_nav.csv
│   └── *.png
├── paper/
│   └── FIN3083_Paper.pdf
├── slides/
│   └── FIN3083_Slides.pdf
└── docs/
    ├── PRD_python.md
    └── Detailed Chinese paper analysis document
```

## How to run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Put the raw CSV files into `data/`

Required files:

```text
data/macro_shibor.csv
data/index_weights.csv
data/stock_prices.csv
data/daily_basic.csv
data/fina_indicator.csv
data/index_prices.csv
```

### 3. Run the full pipeline

```bash
python src/FIN3083_Strategy_Final.py
```

or:

```bash
python src/main.py
```

The code writes CSV files and charts to `output/`.

## Output files

| File | Meaning |
|---|---|
| `performance_summary.csv` | full / in-sample / out-of-sample metrics |
| `regime_sharpe.csv` | performance by Risk-On and Risk-Off regime |
| `ic_stability.csv` | factor IC stability and t-statistics |
| `factor_weights.csv` | dynamic factor weights by year and regime |
| `daily_nav.csv` | daily strategy NAV and benchmark NAV |
| `selected_stocks.csv` | selected top-25 stocks each month |
| `turnover_by_rebalance.csv` | monthly turnover and exposure records |
| `equity_curve.png` | strategy vs CSI 300 NAV |
| `drawdown.png` | drawdown curve |
| `annual_returns.png` | calendar-year returns |
| `return_distribution.png` | daily return histogram |
| `rolling_sharpe.png` | rolling 1-year Sharpe |
| `factor_weights_evolution.png` | factor weight evolution |
| `ic_time_series.png` | IC time series |

## Notes

The repository intentionally ignores raw CSV files in `data/`, because the original data files are large. Keep the six CSV files locally, then run the pipeline. The `output/` folder contains a generated example result.
