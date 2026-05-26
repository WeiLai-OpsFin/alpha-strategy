"""
Configuration for the FIN3083 macro liquidity-driven state-switching
factor rotation strategy.

The default parameters in this file are aligned with the final paper and
presentation deck:
- 120-stock stratified CSI 300 universe: 48 large, 42 mid, 30 small
- 25 stocks selected at each monthly rebalance
- 3-month Shibor median from the in-sample period for regime switching
- 2-year rolling Spearman IC/IR factor weighting
- Three risk overlays: defensive allocation, volatility targeting, loss protection
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd


@dataclass(frozen=True)
class Config:
    """Central project configuration."""

    # ---------------------------------------------------------------------
    # Paths
    # ---------------------------------------------------------------------
    project_path: Path = Path(__file__).resolve().parents[2]

    # ---------------------------------------------------------------------
    # Backtest period
    # ---------------------------------------------------------------------
    start_date: pd.Timestamp = pd.Timestamp("2015-01-01")
    end_date: pd.Timestamp = pd.Timestamp("2024-12-31")
    in_sample_end: pd.Timestamp = pd.Timestamp("2019-12-31")
    out_sample_start: pd.Timestamp = pd.Timestamp("2020-01-01")

    # ---------------------------------------------------------------------
    # Universe and portfolio construction
    # ---------------------------------------------------------------------
    num_stocks_total: int = 120
    large_sample: int = 48
    medium_sample: int = 42
    small_sample: int = 30
    num_stocks_select: int = 25
    random_seed: int = 20260417

    # ---------------------------------------------------------------------
    # Factor / IC parameters
    # ---------------------------------------------------------------------
    trading_days_year: int = 252
    risk_free_rate: float = 0.03
    shibor_window: int = 20  # only used to fill missing shibor_ma20 values
    lowvol_window: int = 20
    future_return_window: int = 20
    ic_window_years: int = 2
    min_ic_threshold: float = 0.005
    min_ic_months: int = 6
    ic_momentum_months: int = 12
    ic_momentum_decline: float = 0.20
    ic_momentum_cut: float = 0.50
    max_factor_weight: float = 0.40

    # ---------------------------------------------------------------------
    # Transaction costs
    # Commission 0.025%, stamp duty 0.10% on sells, impact 0.05% one-way.
    # A full buy-sell round trip is therefore roughly 25 bp.
    # ---------------------------------------------------------------------
    commission_rate: float = 0.00025
    stamp_duty: float = 0.001
    impact_cost: float = 0.0005

    # ---------------------------------------------------------------------
    # Risk management overlays
    # ---------------------------------------------------------------------
    risk_off_equity_exposure: float = 0.50
    volatility_target: float = 0.20
    vol_lookback: int = 60
    vol_scale_min: float = 0.50
    vol_scale_max: float = 1.50
    loss_trigger: float = -0.08
    loss_protection_scale: float = 0.50

    @property
    def data_path(self) -> Path:
        """Directory containing the six raw CSV files."""
        return self.project_path / "data"

    @property
    def output_path(self) -> Path:
        """Output directory. It is created automatically."""
        path = self.project_path / "output"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def total_buy_cost(self) -> float:
        """One-way buy cost."""
        return self.commission_rate + self.impact_cost

    @property
    def total_sell_cost(self) -> float:
        """One-way sell cost, including stamp duty."""
        return self.commission_rate + self.stamp_duty + self.impact_cost

    def get_summary(self) -> dict[str, str | int | float]:
        """Human-readable configuration summary."""
        return {
            "Backtest Period": f"{self.start_date.date()} to {self.end_date.date()}",
            "In-Sample": f"{self.start_date.date()} to {self.in_sample_end.date()}",
            "Out-of-Sample": f"{self.out_sample_start.date()} to {self.end_date.date()}",
            "Universe Size": self.num_stocks_total,
            "Stocks Selected": self.num_stocks_select,
            "Transaction Costs Round Trip": f"{(self.total_buy_cost + self.total_sell_cost) * 10000:.1f} bp",
            "IC Window": f"{self.ic_window_years} years",
            "Volatility Target": f"{self.volatility_target:.0%}",
        }


# -------------------------------------------------------------------------
# Factor definitions
# -------------------------------------------------------------------------
FACTOR_COLUMNS = ["growth", "value", "quality", "lowvol", "dividend", "momentum"]

FACTOR_DESCRIPTIONS = {
    "growth": "Net profit year-over-year growth",
    "value": "Earnings-to-price plus book-to-price",
    "quality": "Return on equity",
    "lowvol": "Inverse 20-day realized volatility",
    "dividend": "Dividend yield",
    "momentum": "12-1 month price momentum",
}

# Fallback weights used when a regime has fewer than 6 months of IC history.
DEFAULT_WEIGHTS = {
    "Risk-On": {
        "growth": 0.05,
        "value": 0.15,
        "quality": 0.05,
        "lowvol": 0.20,
        "dividend": 0.10,
        "momentum": 0.45,
    },
    "Risk-Off": {
        "growth": 0.00,
        "value": 0.20,
        "quality": 0.05,
        "lowvol": 0.35,
        "dividend": 0.20,
        "momentum": 0.20,
    },
}

REGIME_THRESHOLD_QUANTILE = 0.50
