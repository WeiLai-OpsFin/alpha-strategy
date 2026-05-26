from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .config import Config
from .data_import import parse_tushare_date


def _filter_date(df: pd.DataFrame, col: str, config: Config) -> pd.DataFrame:
    return df.loc[df[col].between(config.start_date, config.end_date)].copy()


def clean_data(raw: Dict[str, pd.DataFrame], config: Config) -> Dict[str, pd.DataFrame]:
    """Clean raw data and keep the backtest period."""
    print("[2/8] 清洗数据...")

    shibor = raw["shibor"].copy()
    shibor["trade_date"] = parse_tushare_date(shibor["date"])
    shibor = _filter_date(shibor, "trade_date", config)
    shibor = shibor.sort_values("trade_date")
    shibor["shibor_3m"] = pd.to_numeric(shibor["shibor_3m"], errors="coerce")
    if "shibor_ma20" not in shibor.columns:
        shibor["shibor_ma20"] = np.nan
    shibor["shibor_ma20"] = shibor["shibor_ma20"].fillna(
        shibor["shibor_3m"].rolling(config.shibor_window, min_periods=5).mean()
    )
    shibor = shibor[["trade_date", "shibor_3m", "shibor_ma20"]]

    weights = raw["index_weights"].copy()
    weights["trade_date"] = parse_tushare_date(weights["trade_date"])
    weights["ts_code"] = weights["con_code"]
    weights["weight"] = pd.to_numeric(weights["weight"], errors="coerce")
    weights = _filter_date(weights, "trade_date", config)
    weights = weights[["trade_date", "ts_code", "weight"]].dropna()

    prices = raw["stock_prices"].copy()
    prices["trade_date"] = parse_tushare_date(prices["trade_date"])
    prices = _filter_date(prices, "trade_date", config)
    for col in ["open", "high", "low", "close", "pre_close", "pct_chg", "vol", "amount"]:
        if col in prices.columns:
            prices[col] = pd.to_numeric(prices[col], errors="coerce")
    prices = prices.sort_values(["ts_code", "trade_date"])
    calculated_return = prices.groupby("ts_code")["close"].pct_change()
    prices["daily_return"] = prices["pct_chg"].div(100).where(prices["pct_chg"].notna(), calculated_return)
    prices = prices[["ts_code", "trade_date", "open", "high", "low", "close", "vol", "daily_return"]]
    prices = prices.dropna(subset=["ts_code", "trade_date", "close"])

    index_prices = raw["index_prices"].copy()
    index_prices["trade_date"] = parse_tushare_date(index_prices["trade_date"])
    index_prices = _filter_date(index_prices, "trade_date", config)
    for col in ["close", "index_return", "pct_chg"]:
        if col in index_prices.columns:
            index_prices[col] = pd.to_numeric(index_prices[col], errors="coerce")
    index_prices = index_prices.sort_values("trade_date")
    if "index_return" not in index_prices.columns:
        index_prices["index_return"] = np.nan
    index_prices["index_return"] = index_prices["index_return"].where(
        index_prices["index_return"].notna(), index_prices["close"].pct_change()
    )
    if index_prices["index_return"].abs().max(skipna=True) > 1:
        index_prices["index_return"] = index_prices["index_return"] / 100
    index_prices = index_prices[["trade_date", "close", "index_return"]].rename(columns={"close": "index_close"})

    basic = raw["daily_basic"].copy()
    basic["trade_date"] = parse_tushare_date(basic["trade_date"])
    basic = _filter_date(basic, "trade_date", config)
    for col in ["pe", "pb", "dv_ratio", "turnover_rate", "ep", "bp"]:
        if col in basic.columns:
            basic[col] = pd.to_numeric(basic[col], errors="coerce")
    if "ep" not in basic.columns:
        basic["ep"] = 1 / basic["pe"]
    if "bp" not in basic.columns:
        basic["bp"] = 1 / basic["pb"]
    basic.loc[basic["ep"] < 0, "ep"] = np.nan
    basic.loc[basic["bp"] < 0, "bp"] = np.nan
    basic = basic[["ts_code", "trade_date", "pe", "pb", "ep", "bp", "dv_ratio", "turnover_rate"]]

    fina = raw["fina_indicator"].copy()
    fina["ann_date"] = parse_tushare_date(fina["ann_date"])
    fina["end_date"] = parse_tushare_date(fina["end_date"])
    if "profit_dedt_yoy" not in fina.columns and "dt_netprofit_yoy" in fina.columns:
        fina["profit_dedt_yoy"] = fina["dt_netprofit_yoy"]
    for col in ["roe", "grossprofit_margin", "profit_dedt_yoy", "dt_eps"]:
        if col in fina.columns:
            fina[col] = pd.to_numeric(fina[col], errors="coerce")
    fina = fina.loc[fina["ann_date"].notna() & (fina["ann_date"] <= config.end_date)].copy()
    fina = fina[["ts_code", "ann_date", "end_date", "roe", "grossprofit_margin", "profit_dedt_yoy"]]
    fina = fina.drop_duplicates(["ts_code", "ann_date", "end_date"], keep="last")

    cleaned = {
        "shibor": shibor,
        "index_weights": weights,
        "stock_prices": prices,
        "index_prices": index_prices,
        "daily_basic": basic,
        "fina_indicator": fina,
    }
    for name, df in cleaned.items():
        print(f"  - {name}: {len(df):,} 行")
    return cleaned


def stratified_sample_universe(weights: pd.DataFrame, config: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Select a 120-stock stratified CSI 300 universe using only in-sample data.

    The index weight column is stored in percentage points, for example 4.643
    means 4.643%.  To avoid look-ahead bias, the average weight used for
    stratification is computed only from 2015-2019 in-sample data.  Stocks are
    then divided by rank into large / mid / small buckets and sampled as
    48 / 42 / 30 stocks respectively.
    """
    print("[3/8] 分层抽样股票池...")

    in_sample_weights = weights.loc[weights["trade_date"] <= config.in_sample_end].copy()
    if in_sample_weights.empty:
        in_sample_weights = weights.copy()

    summary = (
        in_sample_weights.groupby("ts_code", as_index=False)
        .agg(avg_weight=("weight", "mean"), observations=("weight", "size"))
        .sort_values("avg_weight", ascending=False)
        .reset_index(drop=True)
    )
    summary["rank"] = np.arange(1, len(summary) + 1)

    # CSI 300-style rank buckets used in the paper/deck:
    # top 120 = large, next 180 = medium, remaining historical constituents = small.
    large_cut = min(120, len(summary))
    medium_cut = min(300, len(summary))
    summary["size_group"] = np.select(
        [summary["rank"] <= large_cut, summary["rank"] <= medium_cut],
        ["Large", "Medium"],
        default="Small",
    )

    targets = {
        "Large": config.large_sample,
        "Medium": config.medium_sample,
        "Small": config.small_sample,
    }
    parts = []
    for offset, (group, target) in enumerate(targets.items()):
        group_df = summary.loc[summary["size_group"] == group]
        n = min(target, len(group_df))
        if n > 0:
            parts.append(group_df.sample(n=n, random_state=config.random_seed + offset))

    if not parts:
        raise ValueError("无法从 index_weights.csv 中抽样股票池，请检查数据。")

    sample = pd.concat(parts, ignore_index=True).drop_duplicates("ts_code")

    if len(sample) < config.num_stocks_total:
        missing = config.num_stocks_total - len(sample)
        extra_pool = summary.loc[~summary["ts_code"].isin(sample["ts_code"])]
        sample = pd.concat([sample, extra_pool.head(missing)], ignore_index=True)

    sample = sample.head(config.num_stocks_total).copy()
    print("  - 使用样本内权重做分层，避免未来函数")
    print("  - 抽样结果:")
    print(sample["size_group"].value_counts().reindex(["Large", "Medium", "Small"]).fillna(0).astype(int).to_string())
    return sample, summary

def keep_universe(cleaned: Dict[str, pd.DataFrame], sample: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    universe = set(sample["ts_code"])
    filtered = dict(cleaned)
    for key in ["stock_prices", "daily_basic", "fina_indicator"]:
        filtered[key] = cleaned[key].loc[cleaned[key]["ts_code"].isin(universe)].copy()
    return filtered
