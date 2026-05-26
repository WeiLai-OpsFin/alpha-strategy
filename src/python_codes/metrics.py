"""
FIN3083 Group Project - Performance Metrics Module

This module calculates comprehensive performance metrics including:
1. Standard risk-adjusted metrics (Sharpe, Sortino, Calmar, etc.)
2. Custom strategy-specific metrics (Turnover, Holding Period, Factor Exposure)
3. Statistical significance tests

Author: FIN3083 Group
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .config import Config, FACTOR_COLUMNS


def _max_drawdown(nav: pd.Series) -> float:
    """Calculate maximum drawdown from NAV series."""
    peak = nav.cummax()
    dd = nav / peak - 1.0
    return float(dd.min())


def _calculate_var(returns: pd.Series, confidence: float = 0.05) -> float:
    """Calculate Value at Risk at given confidence level."""
    return float(returns.quantile(confidence))


def _calculate_cvar(returns: pd.Series, confidence: float = 0.05) -> float:
    """Calculate Conditional Value at Risk (Expected Shortfall)."""
    var = _calculate_var(returns, confidence)
    return float(returns[returns <= var].mean()) if len(returns[returns <= var]) > 0 else var


def _summary_one(df: pd.DataFrame, label: str, config: Config) -> dict[str, float | str | int]:
    """
    Calculate comprehensive performance metrics for a time period.

    Returns standard metrics plus custom metrics required by the project.
    """
    if df.empty:
        return {"sample": label}

    returns = df["net_return"].fillna(0.0)
    bench = df["benchmark_return"].fillna(0.0)
    nav = (1.0 + returns).cumprod()
    n = len(returns)

    # Basic return metrics
    total_return = float(nav.iloc[-1] - 1.0)
    annualized_return = float((1.0 + total_return) ** (config.trading_days_year / max(n, 1)) - 1.0)
    annualized_volatility = float(returns.std(ddof=1) * np.sqrt(config.trading_days_year))

    # Risk-adjusted metrics
    sharpe_ratio = float((annualized_return - config.risk_free_rate) / annualized_volatility) if annualized_volatility > 0 else np.nan

    downside = returns.where(returns < 0, 0.0).std(ddof=1) * np.sqrt(config.trading_days_year)
    sortino_ratio = float((annualized_return - config.risk_free_rate) / downside) if downside > 0 else np.nan

    maximum_drawdown = _max_drawdown(nav)
    calmar_ratio = float(annualized_return / abs(maximum_drawdown)) if maximum_drawdown < 0 else np.nan

    # Active return metrics
    active = returns - bench
    tracking_error = float(active.std(ddof=1) * np.sqrt(config.trading_days_year))
    bench_annual = ((1.0 + bench).prod() ** (config.trading_days_year / max(n, 1)) - 1.0)
    information_ratio = float((annualized_return - bench_annual) / tracking_error) if tracking_error > 0 else np.nan

    win_rate = float((returns > 0).mean())
    profit_factor = float(abs(returns[returns > 0].sum() / returns[returns < 0].sum())) if len(returns[returns < 0]) > 0 else np.inf

    # Tail risk metrics
    var_95 = _calculate_var(returns, 0.05)
    cvar_95 = _calculate_cvar(returns, 0.05)

    # Skewness and Kurtosis
    skewness = float(stats.skew(returns))
    kurtosis = float(stats.kurtosis(returns))

    return {
        "sample": label,
        "start_date": df["date"].min(),
        "end_date": df["date"].max(),
        "days": n,
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "maximum_drawdown": maximum_drawdown,
        "calmar_ratio": calmar_ratio,
        "information_ratio": information_ratio,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "final_nav": float(nav.iloc[-1]),
    }


def calculate_custom_metrics(
    daily_nav: pd.DataFrame,
    trade_results: pd.DataFrame,
    selected: pd.DataFrame,
    config: Config
) -> dict[str, float]:
    """
    Calculate strategy-specific custom metrics.

    Custom Metrics:
    1. Average Holding Period (days)
    2. Average Turnover per Rebalance
    3. Factor Exposure (correlation with market)
    4. Up-Capture vs Down-Capture
    5. Return Attribution by Regime
    """
    custom = {}

    # 1. Average Holding Period
    if not trade_results.empty and "rebalance_date" in trade_results.columns:
        avg_holding_days = trade_results.groupby("rebalance_date").size().mean()
        custom["avg_holding_period_days"] = float(avg_holding_days) if pd.notna(avg_holding_days) else 0.0
    else:
        custom["avg_holding_period_days"] = config.future_return_window

    # 2. Monthly Turnover (estimate)
    if not selected.empty:
        dates = selected["trade_date"].unique()
        if len(dates) > 1:
            turnover_rates = []
            for i in range(1, len(dates)):
                prev_stocks = set(selected[selected["trade_date"] == dates[i-1]]["ts_code"])
                curr_stocks = set(selected[selected["trade_date"] == dates[i]]["ts_code"])
                turnover = len(prev_stocks - curr_stocks) / config.num_stocks_select
                turnover_rates.append(turnover)
            custom["avg_monthly_turnover"] = float(np.mean(turnover_rates)) if turnover_rates else 0.0
        else:
            custom["avg_monthly_turnover"] = 0.0
    else:
        custom["avg_monthly_turnover"] = 0.0

    # 3. Market Beta (correlation with benchmark)
    returns = daily_nav["net_return"].fillna(0.0)
    bench = daily_nav["benchmark_return"].fillna(0.0)
    if len(returns) > 30 and returns.std() > 0 and bench.std() > 0:
        custom["strategy_beta"] = float(returns.cov(bench) / bench.var()) if bench.var() > 0 else 0.0
        custom["correlation_with_market"] = float(returns.corr(bench))
    else:
        custom["strategy_beta"] = 0.0
        custom["correlation_with_market"] = 0.0

    # 4. Up/Down Capture Ratios
    up_months = bench > 0
    down_months = bench < 0

    if up_months.sum() > 0:
        custom["up_capture"] = float(returns[up_months].mean() / bench[up_months].mean()) if bench[up_months].mean() != 0 else 0.0
    else:
        custom["up_capture"] = 0.0

    if down_months.sum() > 0:
        custom["down_capture"] = float(returns[down_months].mean() / bench[down_months].mean()) if bench[down_months].mean() != 0 else 0.0
    else:
        custom["down_capture"] = 0.0

    # 5. Regime Attribution
    if "regime_state" in daily_nav.columns:
        for regime in ["Risk-On", "Risk-Off"]:
            regime_data = daily_nav[daily_nav["regime_state"] == regime]
            if not regime_data.empty:
                regime_return = regime_data["net_return"].mean() * config.trading_days_year
                custom[f"{regime.lower().replace('-', '_')}_annual_return"] = float(regime_return)

    return custom


def calculate_metrics(
    daily_nav: pd.DataFrame,
    ic_daily: pd.DataFrame,
    config: Config,
    trade_results: pd.DataFrame = None,
    selected: pd.DataFrame = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Calculate all performance metrics including standard and custom metrics.

    Returns:
        (summary DataFrame, regime_summary DataFrame, ic_stability DataFrame, custom_metrics DataFrame)
    """
    print("[结果] 计算绩效指标...")
    daily_nav = daily_nav.sort_values("date").copy()

    # Standard performance metrics
    rows = [
        _summary_one(daily_nav, "Full Sample", config),
        _summary_one(daily_nav.loc[daily_nav["date"] <= config.in_sample_end], "In Sample", config),
        _summary_one(daily_nav.loc[daily_nav["date"] >= config.out_sample_start], "Out of Sample", config),
    ]
    summary = pd.DataFrame(rows)

    # Regime-specific metrics
    regime_rows = []
    for regime, df in daily_nav.groupby("regime_state"):
        row = _summary_one(df, f"Regime: {regime}", config)
        row["regime_state"] = regime
        regime_rows.append(row)
    regime_summary = pd.DataFrame(regime_rows)

    # IC stability metrics
    ic_rows = []
    for factor_name in FACTOR_COLUMNS:
        if factor_name in ic_daily.columns:
            ic_series = ic_daily[factor_name].dropna()
            if len(ic_series) > 0:
                ic_rows.append({
                    "factor": factor_name,
                    "mean_ic": ic_series.mean(),
                    "std_ic": ic_series.std(),
                    "ic_stability": ic_series.mean() / ic_series.std() if ic_series.std() not in [0, np.nan] else np.nan,
                    "ic_t_stat": ic_series.mean() / (ic_series.std() / np.sqrt(len(ic_series))) if ic_series.std() > 0 else 0,
                    "positive_ic_ratio": (ic_series > 0).mean(),
                })
    ic_stability = pd.DataFrame(ic_rows)

    # Custom metrics
    custom_dict = calculate_custom_metrics(daily_nav, trade_results, selected, config)
    custom_df = pd.DataFrame([custom_dict])

    # Print summary
    print("\n=== 策略绩效摘要 ===")
    for _, row in summary.iterrows():
        print(f"\n{row['sample']}:")
        print(f"  总收益: {row['total_return']:.2%}")
        print(f"  年化收益: {row['annualized_return']:.2%}")
        print(f"  Sharpe: {row['sharpe_ratio']:.3f}")
        print(f"  最大回撤: {row['maximum_drawdown']:.2%}")

    return summary, regime_summary, ic_stability, custom_df
