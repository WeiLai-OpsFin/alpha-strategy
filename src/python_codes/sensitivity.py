"""
FIN3083 Group Project - Sensitivity Analysis Module

This module performs robustness tests across different parameter configurations:
1. IC window sensitivity
2. Number of stocks selected
3. Transaction cost sensitivity

Author: FIN3083 Group
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import replace

from .config import Config
from .data_import import load_raw_data
from .data_clean import clean_data, stratified_sample_universe, keep_universe
from .macro_regime import build_macro_regime
from .factor_calc import calculate_factors
from .signal_gen import generate_factor_weights, generate_signals
from .backtest import run_backtest
from .metrics import calculate_metrics


def analyze_ic_window_sensitivity(config: Config, windows: list[int] = None) -> pd.DataFrame:
    """Test strategy performance across different IC calculation windows."""
    if windows is None:
        windows = [1, 2, 3, 4, 5]

    print("\n[敏感性分析] IC窗口期测试...")
    results = []

    for window in windows:
        test_config = replace(config, ic_window_years=window)

        try:
            raw = load_raw_data(test_config)
            cleaned = clean_data(raw, test_config)
            sample, _ = stratified_sample_universe(cleaned["index_weights"], test_config)
            cleaned = keep_universe(cleaned, sample)
            calendar = cleaned["stock_prices"][["trade_date"]].drop_duplicates().sort_values("trade_date")
            regime = build_macro_regime(cleaned["shibor"], calendar, test_config)
            factor = calculate_factors(cleaned, regime, test_config)
            factor_weights, ic_daily = generate_factor_weights(factor, test_config)
            selected, _ = generate_signals(factor, factor_weights, test_config)
            daily_nav, trade_results = run_backtest(selected, factor, regime, test_config)
            perf, regime_perf, ic_stab, _ = calculate_metrics(daily_nav, ic_daily, test_config, trade_results, selected)

            full_sample = perf[perf["sample"] == "Full Sample"].iloc[0]
            results.append({
                "ic_window_years": window,
                "total_return": full_sample["total_return"],
                "annualized_return": full_sample["annualized_return"],
                "sharpe_ratio": full_sample["sharpe_ratio"],
                "max_drawdown": full_sample["maximum_drawdown"],
                "calmar_ratio": full_sample["calmar_ratio"],
                "win_rate": full_sample["win_rate"],
            })
            print(f"  IC窗口 {window}年: Sharpe={full_sample['sharpe_ratio']:.3f}, 年化收益={full_sample['annualized_return']:.2%}")

        except Exception as e:
            print(f"  IC窗口 {window}年: 运行失败 - {e}")

    return pd.DataFrame(results)


def analyze_num_stocks_sensitivity(config: Config, stock_counts: list[int] = None) -> pd.DataFrame:
    """Test strategy performance across different number of selected stocks."""
    if stock_counts is None:
        stock_counts = [10, 15, 20, 25, 30]

    print("\n[敏感性分析] 选股数量测试...")
    results = []

    for num_stocks in stock_counts:
        test_config = replace(config, num_stocks_select=num_stocks)

        try:
            raw = load_raw_data(test_config)
            cleaned = clean_data(raw, test_config)
            sample, _ = stratified_sample_universe(cleaned["index_weights"], test_config)
            cleaned = keep_universe(cleaned, sample)
            calendar = cleaned["stock_prices"][["trade_date"]].drop_duplicates().sort_values("trade_date")
            regime = build_macro_regime(cleaned["shibor"], calendar, test_config)
            factor = calculate_factors(cleaned, regime, test_config)
            factor_weights, ic_daily = generate_factor_weights(factor, test_config)
            selected, _ = generate_signals(factor, factor_weights, test_config)
            daily_nav, trade_results = run_backtest(selected, factor, regime, test_config)
            perf, regime_perf, ic_stab, _ = calculate_metrics(daily_nav, ic_daily, test_config, trade_results, selected)

            full_sample = perf[perf["sample"] == "Full Sample"].iloc[0]
            results.append({
                "num_stocks": num_stocks,
                "total_return": full_sample["total_return"],
                "annualized_return": full_sample["annualized_return"],
                "sharpe_ratio": full_sample["sharpe_ratio"],
                "max_drawdown": full_sample["maximum_drawdown"],
                "calmar_ratio": full_sample["calmar_ratio"],
                "win_rate": full_sample["win_rate"],
            })
            print(f"  选股{num_stocks}只: Sharpe={full_sample['sharpe_ratio']:.3f}, 年化收益={full_sample['annualized_return']:.2%}")

        except Exception as e:
            print(f"  选股{num_stocks}只: 运行失败 - {e}")

    return pd.DataFrame(results)


def analyze_transaction_cost_sensitivity(config: Config) -> pd.DataFrame:
    """Test strategy performance across different transaction cost assumptions."""
    print("\n[敏感性分析] 交易成本测试...")

    scenarios = [
        {"name": "Low Cost", "commission": 0.0001, "stamp": 0.001, "impact": 0.0002},
        {"name": "Base Case", "commission": 0.00025, "stamp": 0.001, "impact": 0.0005},
        {"name": "High Cost", "commission": 0.0005, "stamp": 0.001, "impact": 0.001},
        {"name": "Very High", "commission": 0.001, "stamp": 0.001, "impact": 0.002},
    ]

    results = []

    for scenario in scenarios:
        test_config = replace(
            config,
            commission_rate=scenario["commission"],
            stamp_duty=scenario["stamp"],
            impact_cost=scenario["impact"]
        )

        try:
            raw = load_raw_data(test_config)
            cleaned = clean_data(raw, test_config)
            sample, _ = stratified_sample_universe(cleaned["index_weights"], test_config)
            cleaned = keep_universe(cleaned, sample)
            calendar = cleaned["stock_prices"][["trade_date"]].drop_duplicates().sort_values("trade_date")
            regime = build_macro_regime(cleaned["shibor"], calendar, test_config)
            factor = calculate_factors(cleaned, regime, test_config)
            factor_weights, ic_daily = generate_factor_weights(factor, test_config)
            selected, _ = generate_signals(factor, factor_weights, test_config)
            daily_nav, trade_results = run_backtest(selected, factor, regime, test_config)
            perf, regime_perf, ic_stab, _ = calculate_metrics(daily_nav, ic_daily, test_config, trade_results, selected)

            full_sample = perf[perf["sample"] == "Full Sample"].iloc[0]
            total_cost = test_config.total_buy_cost + test_config.total_sell_cost

            results.append({
                "scenario": scenario["name"],
                "total_cost_bp": total_cost * 10000,
                "total_return": full_sample["total_return"],
                "annualized_return": full_sample["annualized_return"],
                "sharpe_ratio": full_sample["sharpe_ratio"],
                "max_drawdown": full_sample["maximum_drawdown"],
            })
            print(f"  {scenario['name']}: 总成本={total_cost*10000:.0f}bp, Sharpe={full_sample['sharpe_ratio']:.3f}")

        except Exception as e:
            print(f"  {scenario['name']}: 运行失败 - {e}")

    return pd.DataFrame(results)


def run_sensitivity_analysis(config: Config | None = None) -> dict[str, pd.DataFrame]:
    """Run comprehensive sensitivity analysis across all parameters."""
    config = config or Config()

    print("=" * 70)
    print("FIN3083 - 参数敏感性分析")
    print("=" * 70)

    results = {}

    results["ic_window"] = analyze_ic_window_sensitivity(config)
    results["num_stocks"] = analyze_num_stocks_sensitivity(config)
    results["transaction_cost"] = analyze_transaction_cost_sensitivity(config)

    out = config.output_path
    for name, df in results.items():
        df.to_csv(out / f"sensitivity_{name}.csv", index=False)

    print("\n" + "=" * 70)
    print("敏感性分析完成！")
    print("=" * 70)

    return results
