from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config, FACTOR_COLUMNS


def _merge_financial_asof(base: pd.DataFrame, fina: pd.DataFrame) -> pd.DataFrame:
    """Attach the latest announced financial report known on each trade date."""
    left = base[["ts_code", "trade_date"]].drop_duplicates().copy()
    right = fina[["ts_code", "ann_date", "roe", "grossprofit_margin", "profit_dedt_yoy"]].dropna(subset=["ts_code", "ann_date"]).copy()

    # merge_asof is much faster than scanning every stock one by one.
    left = left.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    right = right.sort_values(["ann_date", "ts_code"]).reset_index(drop=True)
    merged = pd.merge_asof(
        left,
        right,
        left_on="trade_date",
        right_on="ann_date",
        by="ts_code",
        direction="backward",
    )
    return merged.drop(columns=["ann_date"], errors="ignore")


def _winsorize_and_zscore(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Fast cross-sectional winsorization and z-score by trading day."""
    out = frame.copy()
    grouped = out.groupby("trade_date", sort=False)
    q01 = grouped[columns].quantile(0.01)
    q99 = grouped[columns].quantile(0.99)

    for col in columns:
        values = out[col].astype(float)
        lo = out["trade_date"].map(q01[col])
        hi = out["trade_date"].map(q99[col])
        clipped = values.clip(lower=lo, upper=hi)
        mean_by_date = clipped.groupby(out["trade_date"]).transform("mean")
        std_by_date = clipped.groupby(out["trade_date"]).transform("std").replace(0, np.nan)
        out[col + "_z"] = ((clipped - mean_by_date) / std_by_date).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def _add_rolling_price_factors(prices: pd.DataFrame, config: Config) -> pd.DataFrame:
    """
    Compute price-based factors: future return, low volatility, and momentum.

    Factors calculated:
    - future_return: Return over holding period (for IC calculation)
    - lowvol: Inverse of realized volatility (defensive factor)
    - momentum: 12-1 month price momentum (skip most recent month to avoid reversal)
    """
    rows = []
    for code, g in prices.groupby("ts_code", sort=False):
        g = g.sort_values("trade_date").copy()

        # Future return for IC calculation (holding period return)
        g["future_return"] = g["close"].shift(-config.future_return_window) / g["close"] - 1

        # Low volatility factor: inverse of rolling standard deviation
        rolling_std = g["daily_return"].rolling(config.lowvol_window, min_periods=10).std()
        g["lowvol"] = 1 / rolling_std.replace(0, np.nan)

        # Momentum factor: 12-1 month return (skip most recent month to avoid short-term reversal)
        # Approx 252 trading days in a year, 21 trading days in a month
        g["momentum"] = g["close"].shift(21) / g["close"].shift(252) - 1

        rows.append(g)
    return pd.concat(rows, ignore_index=True)


def calculate_factors(cleaned: dict[str, pd.DataFrame], regime: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Calculate factors, standardize them by date, and add future returns."""
    print("[5/8] 计算因子...", flush=True)
    prices = cleaned["stock_prices"].copy().sort_values(["ts_code", "trade_date"])
    index_prices = cleaned["index_prices"].copy().sort_values("trade_date")
    basic = cleaned["daily_basic"].copy().sort_values(["ts_code", "trade_date"])
    fina = cleaned["fina_indicator"].copy().sort_values(["ts_code", "ann_date"])

    prices = prices.merge(index_prices[["trade_date", "index_return"]], on="trade_date", how="left")
    prices["daily_return"] = prices["daily_return"].fillna(0.0)
    prices["index_return"] = prices["index_return"].fillna(0.0)
    print(f"  - 行情样本: {len(prices):,} 行", flush=True)

    prices = _add_rolling_price_factors(prices, config)
    print("  - 已计算未来收益、低波动、动量", flush=True)

    factor = prices[["ts_code", "trade_date", "close", "daily_return", "index_return", "future_return", "lowvol", "momentum"]].copy()
    factor = factor.merge(basic[["ts_code", "trade_date", "ep", "bp", "dv_ratio", "turnover_rate"]], on=["ts_code", "trade_date"], how="left")
    print("  - 已合并估值数据", flush=True)

    fin_asof = _merge_financial_asof(factor, fina)
    factor = factor.merge(fin_asof, on=["ts_code", "trade_date"], how="left")
    print("  - 已合并财务数据", flush=True)

    factor["growth"] = factor["profit_dedt_yoy"]
    factor["value"] = factor["ep"] + factor["bp"]
    factor["quality"] = factor["roe"]
    factor["dividend"] = factor["dv_ratio"]

    factor = factor.merge(regime[["trade_date", "regime_state"]], on="trade_date", how="left")
    factor["regime_state"] = factor["regime_state"].ffill().bfill()
    factor = _winsorize_and_zscore(factor, FACTOR_COLUMNS)
    print("  - 已做截尾和标准化", flush=True)

    factor["year"] = factor["trade_date"].dt.year
    factor["yearmonth"] = factor["trade_date"].dt.to_period("M").astype(str)

    keep = [
        "ts_code", "trade_date", "year", "yearmonth", "close", "daily_return", "index_return",
        "future_return", "regime_state",
    ] + FACTOR_COLUMNS + [c + "_z" for c in FACTOR_COLUMNS]
    factor = factor[keep]
    print(f"  - 因子表: {len(factor):,} 行", flush=True)
    return factor
