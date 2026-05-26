from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config


def build_macro_regime(shibor: pd.DataFrame, calendar: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Create the daily Risk-On / Risk-Off state from 3-month Shibor.

    Paper rule:
        Risk-On  if Shibor_3M < in-sample median
        Risk-Off if Shibor_3M >= in-sample median

    The threshold is estimated from 2015-2019 only, so the 2020-2024
    out-of-sample period does not leak into the rule.
    """
    print("[4/8] 识别宏观状态...")
    shibor = shibor.sort_values("trade_date").copy()
    shibor["shibor_signal"] = pd.to_numeric(shibor["shibor_3m"], errors="coerce")

    threshold = shibor.loc[
        shibor["trade_date"] <= config.in_sample_end,
        "shibor_signal",
    ].median()
    if np.isnan(threshold):
        threshold = shibor["shibor_signal"].median()

    shibor["regime_state"] = np.where(
        shibor["shibor_signal"] < threshold,
        "Risk-On",
        "Risk-Off",
    )

    calendar = calendar[["trade_date"]].drop_duplicates().sort_values("trade_date")
    regime = pd.merge_asof(
        calendar,
        shibor[["trade_date", "shibor_3m", "shibor_ma20", "shibor_signal", "regime_state"]],
        on="trade_date",
        direction="backward",
    )
    regime = regime.bfill().ffill()
    regime["year"] = regime["trade_date"].dt.year
    regime["month"] = regime["trade_date"].dt.month

    print(f"  - 样本内 Shibor 3M 中位数阈值: {threshold:.4f}")
    print(regime["regime_state"].value_counts().to_string())
    return regime
