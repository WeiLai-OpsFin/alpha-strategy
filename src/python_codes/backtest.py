"""
Backtesting engine for the FIN3083 strategy.

The backtest implements the paper's three sequential risk overlays:
1. Risk-Off defensive allocation: 50% equity / 50% Shibor cash
2. Volatility targeting: 60-day realized vol, target 20%, bounded [0.5, 1.5]
3. Loss protection: after a monthly loss worse than -8%, halve equity next month
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config


def _cash_return_from_shibor(shibor_3m: pd.Series, config: Config) -> pd.Series:
    """Convert annualized 3M Shibor percentages into daily cash returns."""
    return shibor_3m.fillna(config.risk_free_rate * 100).astype(float) / 100.0 / config.trading_days_year


def _vol_scale(previous_returns: list[float], config: Config) -> float:
    """Volatility targeting scale based only on already observed returns."""
    if len(previous_returns) < config.vol_lookback:
        return 1.0
    trailing = pd.Series(previous_returns[-config.vol_lookback:]).dropna()
    realized = trailing.std(ddof=1) * np.sqrt(config.trading_days_year)
    if pd.isna(realized) or realized <= 0:
        return 1.0
    return float(np.clip(config.volatility_target / realized, config.vol_scale_min, config.vol_scale_max))


def _trade_cost_and_turnover(
    prev_holdings: set[str],
    holdings: set[str],
    prev_exposure: float,
    exposure: float,
    config: Config,
) -> tuple[float, float, int, int]:
    """Compute transaction cost and one-way turnover from target weight changes."""
    if not holdings:
        return 0.0, 0.0, 0, 0

    new_weight = {code: exposure / len(holdings) for code in holdings}
    old_weight = {code: prev_exposure / len(prev_holdings) for code in prev_holdings} if prev_holdings else {}

    buy_value = 0.0
    sell_value = 0.0
    for code in set(new_weight) | set(old_weight):
        delta = new_weight.get(code, 0.0) - old_weight.get(code, 0.0)
        if delta > 0:
            buy_value += delta
        elif delta < 0:
            sell_value += -delta

    cost = buy_value * config.total_buy_cost + sell_value * config.total_sell_cost
    turnover = 0.5 * (buy_value + sell_value)
    return float(cost), float(turnover), len(holdings - prev_holdings), len(prev_holdings - holdings)


def run_backtest(
    selected: pd.DataFrame,
    factor: pd.DataFrame,
    regime: pd.DataFrame,
    config: Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run monthly equal-weight top-25 backtest with the three risk overlays."""
    print("[8/8] 执行回测...")

    prices = factor[["ts_code", "trade_date", "daily_return", "index_return"]].copy()
    all_dates = pd.Index(sorted(prices["trade_date"].drop_duplicates()))
    rebalance_dates = sorted(selected["trade_date"].drop_duplicates())

    # Pivot once. This avoids repeatedly scanning the full price table.
    return_matrix = (
        prices.pivot_table(index="trade_date", columns="ts_code", values="daily_return", aggfunc="first")
        .reindex(all_dates)
        .fillna(0.0)
    )
    benchmark_ret = (
        prices.drop_duplicates("trade_date")
        .set_index("trade_date")["index_return"]
        .reindex(all_dates)
        .fillna(0.0)
    )
    regime_daily = regime.set_index("trade_date").reindex(all_dates).ffill().bfill()
    cash_ret = _cash_return_from_shibor(regime_daily["shibor_3m"], config)

    daily_rows: list[dict[str, object]] = []
    trade_rows: list[pd.DataFrame] = []
    turnover_rows: list[dict[str, object]] = []

    prev_holdings: set[str] = set()
    prev_exposure = 0.0
    prev_month_return = 0.0
    realized_returns: list[float] = []
    nav = 1.0
    benchmark_nav = 1.0

    for i, reb_date in enumerate(rebalance_dates):
        current_selection = selected.loc[selected["trade_date"] == reb_date]
        holdings = set(current_selection["ts_code"])
        if not holdings:
            continue

        next_reb = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else all_dates.max()
        interval_dates = all_dates[(all_dates > reb_date) & (all_dates <= next_reb)]
        if len(interval_dates) == 0:
            continue

        reb_regime = current_selection["regime_state"].iloc[0]
        defensive_exposure = config.risk_off_equity_exposure if reb_regime == "Risk-Off" else 1.0
        scale = _vol_scale(realized_returns, config)
        loss_scale = config.loss_protection_scale if prev_month_return < config.loss_trigger else 1.0
        equity_exposure = defensive_exposure * scale * loss_scale
        cash_weight = max(0.0, 1.0 - equity_exposure)

        cost, turnover, buys, sells = _trade_cost_and_turnover(
            prev_holdings, holdings, prev_exposure, equity_exposure, config
        )

        available_holdings = [code for code in holdings if code in return_matrix.columns]
        if available_holdings:
            equity_ret = return_matrix.loc[interval_dates, available_holdings].mean(axis=1).fillna(0.0)
            holding_returns = (1.0 + return_matrix.loc[interval_dates, available_holdings]).prod(axis=0) - 1.0
        else:
            equity_ret = pd.Series(0.0, index=interval_dates)
            holding_returns = pd.Series(dtype=float)

        month_net_returns: list[float] = []
        for j, date in enumerate(interval_dates):
            gross_return = equity_exposure * float(equity_ret.loc[date]) + cash_weight * float(cash_ret.loc[date])
            daily_cost = cost if j == 0 else 0.0
            net_return = gross_return - daily_cost

            nav *= 1.0 + net_return
            benchmark_nav *= 1.0 + float(benchmark_ret.loc[date])
            realized_returns.append(net_return)
            month_net_returns.append(net_return)

            daily_rows.append({
                "date": date,
                "rebalance_date": reb_date,
                "regime_state_at_rebalance": reb_regime,
                "equity_return": float(equity_ret.loc[date]),
                "cash_return": float(cash_ret.loc[date]),
                "portfolio_return": gross_return,
                "transaction_cost": daily_cost,
                "net_return": net_return,
                "nav": nav,
                "benchmark_return": float(benchmark_ret.loc[date]),
                "benchmark_nav": benchmark_nav,
                "equity_exposure": equity_exposure,
                "cash_weight": cash_weight,
                "vol_scale": scale,
                "loss_protection_scale": loss_scale,
            })

        prev_month_return = float(np.prod([1.0 + r for r in month_net_returns]) - 1.0)

        trade_df = holding_returns.reset_index()
        if trade_df.empty:
            trade_df = pd.DataFrame(columns=["ts_code", "holding_return"])
        else:
            trade_df.columns = ["ts_code", "holding_return"]
        trade_df["rebalance_date"] = reb_date
        trade_df["exit_date"] = next_reb
        trade_df["regime_state"] = reb_regime
        trade_df["equity_exposure"] = equity_exposure
        trade_df["net_return_after_cost"] = trade_df["holding_return"] * equity_exposure - cost / max(len(holdings), 1)
        trade_df["is_oos"] = trade_df["rebalance_date"] >= config.out_sample_start
        trade_rows.append(trade_df)

        turnover_rows.append({
            "rebalance_date": reb_date,
            "num_buys": buys,
            "num_sells": sells,
            "turnover": turnover,
            "transaction_cost": cost,
            "regime_state": reb_regime,
            "equity_exposure": equity_exposure,
            "vol_scale": scale,
            "loss_protection_scale": loss_scale,
            "is_oos": reb_date >= config.out_sample_start,
        })

        prev_holdings = holdings
        prev_exposure = equity_exposure

    daily_nav = pd.DataFrame(daily_rows)
    trades = pd.concat(trade_rows, ignore_index=True) if trade_rows else pd.DataFrame()
    turnover_df = pd.DataFrame(turnover_rows)

    if daily_nav.empty:
        return daily_nav, trades

    daily_nav = daily_nav.merge(
        regime[["trade_date", "regime_state"]].rename(columns={"trade_date": "date"}),
        on="date",
        how="left",
    )
    daily_nav["regime_state"] = daily_nav["regime_state"].fillna(daily_nav["regime_state_at_rebalance"])
    daily_nav["is_oos"] = daily_nav["date"] >= config.out_sample_start

    print(f"  - 每日净值: {len(daily_nav):,} 行")
    print(f"  - 持仓交易记录: {len(trades):,} 行")
    if not turnover_df.empty:
        print(f"  - 平均月度换手率: {turnover_df['turnover'].mean():.2%}")
        turnover_df.to_csv(config.output_path / "turnover_by_rebalance.csv", index=False)

    return daily_nav, trades
