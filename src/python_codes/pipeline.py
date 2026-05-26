"""
FIN3083 Group Project - Main Pipeline Module

This module orchestrates the complete backtesting workflow:
1. Data loading and cleaning
2. Universe selection (stratified sampling)
3. Macro regime detection
4. Factor calculation
5. Signal generation with dynamic weight optimization
6. Backtesting with transaction costs
7. Performance metrics calculation
8. Visualization and output generation

Author: FIN3083 Group
"""
from __future__ import annotations

import warnings
import gc
warnings.filterwarnings('ignore')

from .config import Config, FACTOR_DESCRIPTIONS
from .data_import import load_raw_data
from .data_clean import clean_data, keep_universe, stratified_sample_universe
from .macro_regime import build_macro_regime
from .factor_calc import calculate_factors
from .signal_gen import generate_factor_weights, generate_signals
from .backtest import run_backtest
from .metrics import calculate_metrics
from .visualization import make_charts


def run_pipeline(config: Config | None = None) -> None:
    """
    Execute the complete backtesting pipeline.

    Args:
        config: Configuration object. If None, uses default Config().
    """
    config = config or Config()
    config.output_path.mkdir(parents=True, exist_ok=True)

    # Print configuration summary
    print("=" * 70)
    print("FIN3083 Group Project - Macro Liquidity-Driven Factor Rotation Strategy")
    print("=" * 70)
    print(f"\n项目路径: {config.project_path}")
    print(f"回测期间: {config.start_date.date()} 至 {config.end_date.date()}")
    print(f"样本内: {config.start_date.date()} 至 {config.in_sample_end.date()}")
    print(f"样本外: {config.out_sample_start.date()} 至 {config.end_date.date()}")
    print(f"股票池: {config.num_stocks_total}只 (沪深300分层抽样)")
    print(f"选股数量: 每月{config.num_stocks_select}只")
    print(f"因子配置: {', '.join(FACTOR_DESCRIPTIONS.keys())}")
    print("=" * 70)

    # Step 1: Load raw data
    raw = load_raw_data(config)

    # Step 2: Clean data
    cleaned = clean_data(raw, config)

    # Step 3: Universe selection
    sample, universe_summary = stratified_sample_universe(cleaned["index_weights"], config)
    cleaned = keep_universe(cleaned, sample)

    # Step 4: Build macro regime indicator
    calendar = cleaned["stock_prices"][["trade_date"]].drop_duplicates().sort_values("trade_date")
    regime = build_macro_regime(cleaned["shibor"], calendar, config)

    # Step 5: Calculate factors
    factor = calculate_factors(cleaned, regime, config)

    # The raw CSV tables are large.  Release them as soon as factors are built.
    del raw, cleaned
    gc.collect()

    # Step 6: Generate factor weights (dynamic optimization)
    factor_weights, ic_daily = generate_factor_weights(factor, config)

    # Step 7: Generate signals (stock selection)
    selected, all_month_end_scores = generate_signals(factor, factor_weights, config)

    # Step 8: Run backtest
    daily_nav, trade_results = run_backtest(selected, factor, regime, config)

    # Step 9: Calculate metrics
    performance_summary, regime_summary, ic_stability, custom_metrics = calculate_metrics(
        daily_nav, ic_daily, config, trade_results, selected
    )

    # Save outputs before charting.  Then release large raw/factor tables so
    # matplotlib has enough memory on normal laptops and CI runners.
    out = config.output_path
    sample.to_csv(out / "sampled_universe.csv", index=False)
    universe_summary.to_csv(out / "universe_weight_summary.csv", index=False)
    regime.to_csv(out / "macro_regime.csv", index=False)
    factor_weights.to_csv(out / "factor_weights.csv", index=False)
    ic_daily.to_csv(out / "factor_ic_daily.csv", index=False)
    selected.to_csv(out / "selected_stocks.csv", index=False)
    all_month_end_scores.to_csv(out / "month_end_scores.csv", index=False)
    trade_results.to_csv(out / "trade_results.csv", index=False)
    daily_nav.to_csv(out / "daily_nav.csv", index=False)
    performance_summary.to_csv(out / "performance_summary.csv", index=False)
    regime_summary.to_csv(out / "regime_sharpe.csv", index=False)
    ic_stability.to_csv(out / "ic_stability.csv", index=False)
    custom_metrics.to_csv(out / "custom_metrics.csv", index=False)

    del factor, all_month_end_scores, trade_results
    gc.collect()

    # Step 10: Generate visualizations
    make_charts(daily_nav, performance_summary, regime_summary, factor_weights, ic_daily, config)

    # Print final summary
    print("\n" + "=" * 70)
    print("回测完成！输出文件保存在 output/ 文件夹")
    print("=" * 70)
    print("\n关键结果文件:")
    print("  - performance_summary.csv      : 总体绩效指标")
    print("  - regime_sharpe.csv            : 宏观状态绩效")
    print("  - ic_stability.csv             : 因子IC稳定性")
    print("  - custom_metrics.csv           : 自定义指标")
    print("  - daily_nav.csv                : 每日净值")
    print("  - selected_stocks.csv          : 每月选股结果")
    print("  - factor_weights.csv           : 动态因子权重")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()
