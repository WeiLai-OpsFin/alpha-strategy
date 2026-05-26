"""
Signal generation for the FIN3083 factor rotation strategy.

This module follows the final paper/deck methodology:
- Month-end cross-sectional Spearman IC over the full 120-stock universe
- Two-year rolling IC/IR factor weighting by macro regime
- IC threshold, IC momentum filter, negative-IC removal, and 40% cap
- Equal-weighted top-25 stock selection each month
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config, FACTOR_COLUMNS, DEFAULT_WEIGHTS


def month_end_dates(factor: pd.DataFrame) -> pd.DataFrame:
    """Return the actual last trading date in each month."""
    return (
        factor[["trade_date", "yearmonth"]]
        .drop_duplicates()
        .sort_values("trade_date")
        .groupby("yearmonth", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )


def _calc_ic_table(factor: pd.DataFrame) -> pd.DataFrame:
    """Calculate month-end Spearman rank IC for each factor.

    IC is computed across the full stock universe on each month-end date.  We do
    not split the cross-section by regime because regime is a market-wide state
    shared by all stocks on the same date.  The regime label is attached to the
    month-end row and used later when rolling weights are estimated.
    """
    records: list[dict[str, object]] = []
    z_cols = [f"{c}_z" for c in FACTOR_COLUMNS]
    mends = month_end_dates(factor)
    data = factor.merge(mends[["trade_date"]], on="trade_date", how="inner")
    data = data.dropna(subset=["future_return", "regime_state"])

    for date, group in data.groupby("trade_date", sort=False):
        row: dict[str, object] = {
            "trade_date": date,
            "regime_state": group["regime_state"].iloc[0],
        }
        y = group["future_return"]
        for factor_name, z_col in zip(FACTOR_COLUMNS, z_cols):
            valid = pd.concat([group[z_col], y], axis=1).dropna()
            if len(valid) >= 10 and valid[z_col].nunique() > 1 and valid["future_return"].nunique() > 1:
                row[factor_name] = valid[z_col].corr(valid["future_return"], method="spearman")
            else:
                row[factor_name] = np.nan
        records.append(row)

    return pd.DataFrame(records).sort_values("trade_date").reset_index(drop=True)


def _momentum_filter_multiplier(ic_hist: pd.DataFrame, factor_name: str, config: Config) -> float:
    """Return 0.5 when recent IC drops by more than 20%; otherwise 1.0."""
    series = ic_hist[["trade_date", factor_name]].dropna().sort_values("trade_date")[factor_name]
    m = config.ic_momentum_months
    if len(series) < 2 * m:
        return 1.0
    recent = series.iloc[-m:].mean()
    previous = series.iloc[-2 * m:-m].mean()
    if pd.notna(previous) and previous > 0 and recent < previous * (1 - config.ic_momentum_decline):
        return config.ic_momentum_cut
    return 1.0


def _raw_ic_ir_weight(ic_hist: pd.DataFrame, factor_name: str, config: Config) -> float:
    """IC/IR raw weight: max(0, IC) * max(0, IC / std(IC))."""
    series = ic_hist[factor_name].dropna()
    if len(series) < config.min_ic_months:
        return 0.0

    mean_ic = series.mean()
    std_ic = series.std(ddof=1)
    if pd.isna(mean_ic) or mean_ic < config.min_ic_threshold:
        return 0.0
    if pd.isna(std_ic) or std_ic <= 0:
        raw = max(0.0, mean_ic)
    else:
        raw = max(0.0, mean_ic) * max(0.0, mean_ic / std_ic)
    return raw * _momentum_filter_multiplier(ic_hist, factor_name, config)


def _normalize_with_cap(raw: dict[str, float], fallback: dict[str, float], max_weight: float) -> dict[str, float]:
    """Normalize positive raw weights to sum to 1 with a max-weight cap.

    If only one factor has a positive signal, the cap is relaxed; otherwise the
    portfolio would be forced into negative/no-signal factors.
    """
    positive = {k: max(0.0, float(v)) for k, v in raw.items() if pd.notna(v) and v > 0}
    if not positive:
        return fallback.copy()
    if len(positive) == 1:
        only = next(iter(positive))
        return {k: 1.0 if k == only else 0.0 for k in raw}

    total = sum(positive.values())
    weights = {k: positive.get(k, 0.0) / total for k in raw}

    capped: dict[str, float] = {}
    uncapped = set(weights)
    remaining = 1.0
    raw_remaining = {k: positive.get(k, 0.0) for k in weights}

    # Water-filling: cap factors above max_weight and redistribute the rest.
    while uncapped:
        raw_sum = sum(raw_remaining[k] for k in uncapped)
        if raw_sum <= 0:
            equal = remaining / len(uncapped)
            for k in uncapped:
                capped[k] = equal
            break

        tentative = {k: remaining * raw_remaining[k] / raw_sum for k in uncapped}
        over = {k for k, v in tentative.items() if v > max_weight}
        if not over:
            capped.update(tentative)
            break
        for k in over:
            capped[k] = max_weight
            remaining -= max_weight
            uncapped.remove(k)

    final = {k: capped.get(k, 0.0) for k in raw}
    final_sum = sum(final.values())
    if final_sum <= 0:
        return fallback.copy()
    return {k: v / final_sum for k, v in final.items()}


def generate_factor_weights(factor: pd.DataFrame, config: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate dynamic factor weights using rolling Spearman IC/IR."""
    print("[6/8] 计算滚动 Spearman IC 和动态因子权重...")
    ic_daily = _calc_ic_table(factor)
    weight_rows: list[dict[str, object]] = []

    years = sorted(factor["year"].dropna().astype(int).unique())
    for year in years:
        start = pd.Timestamp(f"{year - config.ic_window_years}-01-01")
        end = pd.Timestamp(f"{year - 1}-12-31")
        hist = ic_daily.loc[ic_daily["trade_date"].between(start, end)].copy()

        for regime in ["Risk-On", "Risk-Off"]:
            fallback = DEFAULT_WEIGHTS[regime]
            regime_hist = hist.loc[hist["regime_state"] == regime].copy()

            if len(regime_hist) < config.min_ic_months:
                weights = fallback.copy()
                method = "fallback"
            else:
                raw = {
                    factor_name: _raw_ic_ir_weight(regime_hist, factor_name, config)
                    for factor_name in FACTOR_COLUMNS
                }
                weights = _normalize_with_cap(raw, fallback, config.max_factor_weight)
                method = "ic_ir"

            for factor_name in FACTOR_COLUMNS:
                weight_rows.append({
                    "year": year,
                    "regime_state": regime,
                    "factor": factor_name,
                    "weight": weights.get(factor_name, 0.0),
                    "method": method,
                })

    factor_weights = pd.DataFrame(weight_rows)
    print(f"  - IC记录: {len(ic_daily):,} 个月")
    print(f"  - 权重记录: {len(factor_weights):,} 行")
    return factor_weights, ic_daily


def generate_signals(
    factor: pd.DataFrame,
    factor_weights: pd.DataFrame,
    config: Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create month-end composite scores and select the top N stocks."""
    print("[7/8] 生成月度选股信号...")
    month_ends = month_end_dates(factor)
    signal_base = factor.merge(month_ends[["trade_date"]], on="trade_date", how="inner").copy()

    weight_wide = factor_weights.pivot_table(
        index=["year", "regime_state"],
        columns="factor",
        values="weight",
        aggfunc="first",
    ).reset_index()

    signal_base = signal_base.merge(
        weight_wide,
        on=["year", "regime_state"],
        how="left",
        suffixes=("", "_weight"),
    )

    score = pd.Series(0.0, index=signal_base.index)
    for factor_name in FACTOR_COLUMNS:
        weight_col = f"{factor_name}_weight"
        if weight_col not in signal_base.columns:
            # This should not happen, but keep a clear fallback for debugging.
            weight_col = factor_name
        score += signal_base[f"{factor_name}_z"].fillna(0.0) * signal_base[weight_col].fillna(0.0)
    signal_base["composite_score"] = score

    selected = (
        signal_base.dropna(subset=["composite_score"])
        .sort_values(["trade_date", "composite_score"], ascending=[True, False])
        .groupby("trade_date", group_keys=False)
        .head(config.num_stocks_select)
        .copy()
    )
    selected["target_weight"] = 1.0 / config.num_stocks_select
    selected["rank"] = selected.groupby("trade_date")["composite_score"].rank(
        method="first",
        ascending=False,
    ).astype(int)

    selected_cols = [
        "trade_date", "yearmonth", "regime_state", "ts_code",
        "rank", "composite_score", "target_weight",
    ]
    selected = selected[selected_cols].sort_values(["trade_date", "rank"])

    print(f"  - 选股记录: {len(selected):,} 行；调仓次数: {selected['trade_date'].nunique():,}")
    avg_score = selected.groupby("trade_date")["composite_score"].mean()
    print(f"  - 平均综合得分: {avg_score.mean():.4f} +/- {avg_score.std():.4f}")
    return selected, signal_base
