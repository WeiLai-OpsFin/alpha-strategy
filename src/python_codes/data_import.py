from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import pandas as pd

from .config import Config


def parse_tushare_date(series: pd.Series) -> pd.Series:
    """Parse dates stored as either 20241231 or 2024-12-31."""
    text = series.astype(str).str.replace(r"\.0$", "", regex=True)
    numeric_mask = text.str.fullmatch(r"\d{8}")
    out = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    out.loc[numeric_mask] = pd.to_datetime(text.loc[numeric_mask], format="%Y%m%d", errors="coerce")
    out.loc[~numeric_mask] = pd.to_datetime(text.loc[~numeric_mask], errors="coerce")
    return out


def _existing_usecols(path: Path, desired: Iterable[str]) -> list[str]:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    return [c for c in desired if c in header]


def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"找不到数据文件: {path}")
    return pd.read_csv(path, **kwargs)


def _read_large_by_date(path: Path, usecols: list[str], date_col: str, config: Config, chunksize: int = 300_000) -> pd.DataFrame:
    """Read only needed columns and needed dates from a large CSV."""
    start_int = int(config.start_date.strftime("%Y%m%d"))
    end_int = int(config.end_date.strftime("%Y%m%d"))
    parts = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        numeric_date = pd.to_numeric(chunk[date_col], errors="coerce")
        mask = numeric_date.between(start_int, end_int)
        if mask.any():
            parts.append(chunk.loc[mask].copy())
    if not parts:
        return pd.DataFrame(columns=usecols)
    return pd.concat(parts, ignore_index=True)


def load_raw_data(config: Config) -> Dict[str, pd.DataFrame]:
    """Load every required CSV file with only the columns needed by the strategy."""
    data_path = config.data_path
    print("[1/8] 读取 CSV 数据...", flush=True)

    shibor_path = data_path / "macro_shibor.csv"
    weight_path = data_path / "index_weights.csv"
    basic_path = data_path / "daily_basic.csv"
    fina_path = data_path / "fina_indicator.csv"
    price_path = data_path / "stock_prices.csv"
    index_path = data_path / "index_prices.csv"

    raw = {
        "shibor": _read_csv(shibor_path, usecols=_existing_usecols(shibor_path, ["date", "shibor_3m", "shibor_ma20"])),
        "index_weights": _read_large_by_date(weight_path, _existing_usecols(weight_path, ["index_code", "con_code", "trade_date", "weight"]), "trade_date", config),
        "daily_basic": _read_large_by_date(basic_path, _existing_usecols(basic_path, ["ts_code", "trade_date", "pe", "pb", "dv_ratio", "turnover_rate", "ep", "bp"]), "trade_date", config),
        "fina_indicator": _read_csv(fina_path, usecols=_existing_usecols(fina_path, ["ts_code", "ann_date", "end_date", "roe", "grossprofit_margin", "profit_dedt_yoy", "dt_netprofit_yoy", "dt_eps"])),
        "stock_prices": _read_large_by_date(price_path, _existing_usecols(price_path, ["ts_code", "trade_date", "open", "high", "low", "close", "pct_chg", "vol"]), "trade_date", config),
        "index_prices": _read_large_by_date(index_path, _existing_usecols(index_path, ["ts_code", "trade_date", "close", "index_return", "pct_chg"]), "trade_date", config),
    }

    for name, df in raw.items():
        print(f"  - {name}: {len(df):,} 行", flush=True)
    return raw
