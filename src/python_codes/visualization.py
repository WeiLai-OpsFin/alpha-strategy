"""Visualization helpers for strategy outputs."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import Config, FACTOR_COLUMNS


def _save(fig, path, dpi: int = 150) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def make_charts(
    daily_nav: pd.DataFrame,
    performance_summary: pd.DataFrame,
    regime_summary: pd.DataFrame,
    factor_weights: pd.DataFrame,
    ic_daily: pd.DataFrame,
    config: Config,
) -> None:
    """Generate all charts used in the paper and slides."""
    print("[结果] 生成图表...", flush=True)
    out = config.output_path
    df = daily_nav.sort_values("date").copy()
    if df.empty:
        print("  警告: 无净值数据，跳过图表生成")
        return

    # 1. Equity curve
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["date"], df["nav"], label="Strategy NAV", linewidth=1.3)
    ax.plot(df["date"], df["benchmark_nav"], label="CSI 300 NAV", linewidth=1.2, alpha=0.85)
    if "regime_state" in df.columns:
        risk_on = df[df["regime_state"] == "Risk-On"]
        if not risk_on.empty:
            ax.scatter(risk_on["date"], [0.85] * len(risk_on), alpha=0.12, s=2, label="Risk-On Period")
    ax.set_title("Strategy Equity Curve vs CSI 300 Benchmark")
    ax.set_xlabel("Date")
    ax.set_ylabel("Net Asset Value")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    ax.set_ylim(bottom=0.7)
    _save(fig, out / "equity_curve.png")

    # 2. Drawdown
    drawdown = df["nav"] / df["nav"].cummax() - 1.0
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.fill_between(df["date"], drawdown, 0, alpha=0.28)
    ax.plot(df["date"], drawdown, linewidth=1.0)
    ax.axhline(drawdown.min(), linestyle="--", linewidth=1.0, label=f"Max DD: {drawdown.min():.1%}")
    ax.set_title("Strategy Drawdown Analysis")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save(fig, out / "drawdown.png")

    # 3. Daily return distribution
    returns = df["net_return"].replace([np.inf, -np.inf], np.nan).dropna()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(returns, bins=50, edgecolor="white", alpha=0.75)
    ax.axvline(returns.mean(), linestyle="--", linewidth=1.5, label=f"Mean: {returns.mean():.3f}")
    ax.axvline(0, linewidth=0.8)
    ax.set_title("Daily Return Distribution")
    ax.set_xlabel("Daily Return")
    ax.set_ylabel("Frequency")
    ax.legend()
    _save(fig, out / "return_distribution.png")

    # 4. Annual returns
    annual = df.assign(year=df["date"].dt.year).groupby("year")["net_return"].apply(lambda x: (1.0 + x).prod() - 1.0)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["green" if r >= 0 else "red" for r in annual.values]
    ax.bar(annual.index.astype(str), annual.values, color=colors, alpha=0.75)
    ax.axhline(0, linewidth=0.8)
    ax.set_title("Annual Returns")
    ax.set_xlabel("Year")
    ax.set_ylabel("Return")
    for i, ret in enumerate(annual.values):
        ax.text(i, ret + (0.02 if ret >= 0 else -0.04), f"{ret:.1%}", ha="center", fontsize=8)
    _save(fig, out / "annual_returns.png")

    # 5. Rolling Sharpe
    rolling_mean = df["net_return"].rolling(252, min_periods=60).mean()
    rolling_std = df["net_return"].rolling(252, min_periods=60).std()
    rolling_sharpe = (rolling_mean * config.trading_days_year - config.risk_free_rate) / (rolling_std * np.sqrt(config.trading_days_year))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["date"], rolling_sharpe, linewidth=1.2)
    ax.axhline(0, linestyle="--", linewidth=1.0)
    ax.axhline(1, linestyle="--", linewidth=1.0, label="Sharpe = 1")
    ax.axhline(-1, linestyle="--", linewidth=1.0, label="Sharpe = -1")
    ax.set_title("Rolling 1-Year Sharpe Ratio")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sharpe Ratio")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save(fig, out / "rolling_sharpe.png")

    # 6. Regime performance
    if not regime_summary.empty and "annualized_return" in regime_summary.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["green" if r >= 0 else "red" for r in regime_summary["annualized_return"]]
        ax.bar(regime_summary["regime_state"].astype(str), regime_summary["annualized_return"], color=colors, alpha=0.75)
        ax.axhline(0, linewidth=0.8)
        ax.set_title("Annualized Return by Regime")
        ax.set_xlabel("Regime")
        ax.set_ylabel("Annualized Return")
        for i, ret in enumerate(regime_summary["annualized_return"]):
            ax.text(i, ret + (0.005 if ret >= 0 else -0.01), f"{ret:.2%}", ha="center", fontsize=9)
        _save(fig, out / "regime_performance.png")

    # 7. Factor weights evolution
    if not factor_weights.empty and "year" in factor_weights.columns:
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        for ax, regime in zip(axes, ["Risk-On", "Risk-Off"]):
            data = factor_weights[factor_weights["regime_state"] == regime]
            for factor in FACTOR_COLUMNS:
                sub = data[data["factor"] == factor]
                if not sub.empty:
                    ax.plot(sub["year"], sub["weight"], marker="o", linewidth=1.2, label=factor)
            ax.set_title(f"Factor Weights Evolution - {regime}")
            ax.set_ylabel("Weight")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper left", fontsize=8, ncol=3)
            ax.set_ylim(-0.05, 0.65)
        axes[-1].set_xlabel("Year")
        _save(fig, out / "factor_weights_evolution.png")

    # 8. IC time series
    if not ic_daily.empty and "trade_date" in ic_daily.columns:
        fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=True)
        axes = axes.flatten()
        for ax, factor in zip(axes, FACTOR_COLUMNS):
            if factor in ic_daily.columns:
                ax.plot(ic_daily["trade_date"], ic_daily[factor], linewidth=0.8, alpha=0.75)
                ax.axhline(0, linewidth=0.6)
                mean_ic = ic_daily[factor].mean()
                ax.axhline(mean_ic, linestyle="--", linewidth=1.0, label=f"Mean: {mean_ic:.3f}")
                ax.set_title(f"{factor.upper()} IC Series")
                ax.set_ylabel("IC")
                ax.grid(True, alpha=0.25)
                ax.legend(fontsize=7)
        _save(fig, out / "ic_time_series.png")

    # 9. Cumulative returns by regime
    if "regime_state" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        for regime_name in df["regime_state"].dropna().unique():
            sub = df[df["regime_state"] == regime_name].copy()
            sub["cumulative_return"] = (1.0 + sub["net_return"]).cumprod() - 1.0
            ax.plot(sub["date"], sub["cumulative_return"], label=str(regime_name), linewidth=1.2)
        ax.set_title("Cumulative Returns by Regime")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cumulative Return")
        ax.grid(True, alpha=0.3)
        ax.legend()
        _save(fig, out / "cumulative_by_regime.png")

    print(f"  已生成图表保存至 {out}", flush=True)
