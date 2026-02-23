import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


# ---------- core series helpers ----------

def get_returns(df: pd.DataFrame, profit_col: str) -> pd.Series:
    return pd.to_numeric(df[profit_col], errors="coerce").astype(float)


def compute_equity_curve(df: pd.DataFrame, profit_col: str) -> pd.Series:
    p = get_returns(df, profit_col).fillna(0.0)
    return (1.0 + p).cumprod()


def max_drawdown_from_nav(nav: pd.Series) -> float:
    nav = pd.to_numeric(nav, errors="coerce").dropna().astype(float)
    if nav.empty:
        return np.nan
    running_max = nav.cummax()
    dd = nav / running_max - 1.0
    return float(dd.min())


# ---------- risk/return ratios (annualized, with RF) ----------

def daily_rf_simple_divide(risk_free_rate_annual: float, trading_days: int) -> float:
    # matches your notebook: rf_annual / 252
    return float(risk_free_rate_annual) / float(trading_days)


def sharpe_annualized_excess(
    p: pd.Series,
    risk_free_rate_annual: float = 0.04,
    trading_days: int = 252,
) -> float:
    p = pd.to_numeric(p, errors="coerce").dropna().astype(float)
    if len(p) < 2:
        return np.nan
    rf_daily = daily_rf_simple_divide(risk_free_rate_annual, trading_days)
    ex = p - rf_daily
    sd = ex.std(ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return np.nan
    return float(np.sqrt(trading_days) * (ex.mean() / sd))


def sortino_annualized_excess(
    p: pd.Series,
    risk_free_rate_annual: float = 0.04,
    trading_days: int = 252,
) -> float:
    p = pd.to_numeric(p, errors="coerce").dropna().astype(float)
    if len(p) < 2:
        return np.nan
    rf_daily = daily_rf_simple_divide(risk_free_rate_annual, trading_days)
    expected = p.mean() - rf_daily
    downside = p[p < 0]
    downside_dev = float(np.std(downside))  # ddof=0 like your notebook
    if downside_dev == 0 or not np.isfinite(downside_dev):
        return np.nan
    return float(np.sqrt(trading_days) * (expected / downside_dev))


# ---------- signal-based metrics ----------

def exposure_rate(signal: pd.Series) -> float:
    s = pd.to_numeric(signal, errors="coerce").fillna(0.0)
    return float((s != 0).mean())


def signal_turnover_yearly(signal: pd.Series, trading_days: int = 252) -> float:
    """
    Annualized signal turnover:
      yearly = (sum(|Δsignal|) / N) * trading_days
    For {-1,0,1} state signals, measures how frequently position changes.
    """
    s = pd.to_numeric(signal, errors="coerce").fillna(0.0).astype(float)
    n = len(s)
    if n < 2:
        return np.nan
    total = float(s.diff().abs().sum())
    daily = total / n
    return float(daily * trading_days)


def win_rate_all_days(p: pd.Series) -> float:
    p = pd.to_numeric(p, errors="coerce").dropna().astype(float)
    if len(p) == 0:
        return np.nan
    return float((p > 0).mean())


def win_rate_trade_days(df: pd.DataFrame, profit_col: str, signal_col: str = "signal2") -> float:
    if signal_col not in df.columns or profit_col not in df.columns:
        return np.nan
    sig = pd.to_numeric(df[signal_col], errors="coerce").fillna(0.0)
    p = pd.to_numeric(df[profit_col], errors="coerce")
    mask = sig != 0
    p = p[mask].dropna().astype(float)
    if len(p) == 0:
        return np.nan
    return float((p > 0).mean())


def trade_attempt_rate(df: pd.DataFrame, signal_col: str = "signal2") -> float:
    if signal_col not in df.columns or len(df) == 0:
        return np.nan
    sig = pd.to_numeric(df[signal_col], errors="coerce").fillna(0.0)
    return float((sig != 0).mean())


# ============ mean reversion diagnostics ============

def compute_half_life(series: pd.Series, min_obs: int = 20) -> float:
    """
    Half-life via OLS on: Δx_t = beta * x_{t-1} + eps
    half_life = ln(2) / |beta|
    """
    s = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if len(s) < min_obs:
        return np.nan

    lagged = s.shift(1).dropna()
    delta = s.diff().dropna()
    delta = delta.loc[lagged.index]

    X = lagged.values.reshape(-1, 1)
    y = delta.values

    beta = np.linalg.lstsq(X, y, rcond=None)[0][0]
    if not np.isfinite(beta):
        return np.nan
    if beta == 0:
        return np.inf
    return float(np.log(2) / abs(beta))


def zscore(series: pd.Series, window: int = 5) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    m = s.rolling(window=window).mean()
    sd = s.rolling(window=window).std()
    return (s - m) / sd


def stationarity_diagnostics(
    df: pd.DataFrame,
    price_col: str = "price",
    zscore_window: int = 5,
    min_obs: int = 20,
) -> dict:
    """
    Returns dict for 3 transformations:
      - raw
      - diff (first difference)
      - zscore (rolling z-score)
    Each contains: adf_stat, p_value, half_life, n_obs.
    """
    out = {
        "raw": {"adf_stat": np.nan, "p_value": np.nan, "half_life": np.nan, "n_obs": 0},
        "diff": {"adf_stat": np.nan, "p_value": np.nan, "half_life": np.nan, "n_obs": 0},
        "zscore": {"adf_stat": np.nan, "p_value": np.nan, "half_life": np.nan, "n_obs": 0},
        "note": None,
    }

    if price_col not in df.columns:
        out["note"] = f"missing column '{price_col}'"
        return out

    s_raw = pd.to_numeric(df[price_col], errors="coerce").dropna().astype(float)
    if len(s_raw) < min_obs:
        out["note"] = f"insufficient obs (<{min_obs})"
        out["raw"]["n_obs"] = int(len(s_raw))
        return out

    transforms = {
        "raw": s_raw,
        "diff": s_raw.diff().dropna(),
        "zscore": zscore(s_raw, window=zscore_window).dropna(),
    }

    if adfuller is None:
        out["note"] = "statsmodels not installed; ADF skipped"
        for k, s in transforms.items():
            out[k]["n_obs"] = int(len(s))
            out[k]["half_life"] = compute_half_life(s, min_obs=min_obs)
        return out

    for name, s in transforms.items():
        s = pd.to_numeric(s, errors="coerce").dropna().astype(float)
        out[name]["n_obs"] = int(len(s))
        if len(s) < min_obs:
            continue

        try:
            adf_stat, p_value, *_ = adfuller(s)
            out[name]["adf_stat"] = float(adf_stat)
            out[name]["p_value"] = float(p_value)
        except Exception:
            # keep NaNs; don't break pipeline
            pass

        out[name]["half_life"] = compute_half_life(s, min_obs=min_obs)

    return out



# ---------- aggregator (what you asked for) ----------

def full_portfolio_metrics(
    df: pd.DataFrame,
    profit_col: str = "profit_entry@t1_exit@t2",
    risk_free_rate_annual: float = 0.04,
    trading_days: int = 252,
    signal_col: str = "signal2",
) -> tuple[dict, pd.Series]:
    """
    Aggregate metrics + NAV output (old style).
    Uses:
      - annualized Sharpe/Sortino with RF (matches your calculate_portfolio_performance)
      - max drawdown from NAV
      - turnover annualized (signal-based)
      - win rate (both all-days and trade-days)
    Returns:
      metrics dict
      nav series
    """
    if profit_col not in df.columns:
        raise ValueError(f"profit_col '{profit_col}' not found in df")

    p = get_returns(df, profit_col)
    nav = compute_equity_curve(df, profit_col)

    metrics = {
        # basic return stats
        "mean_daily_return": float(p.mean()),
        "std_daily_return": float(p.std(ddof=1)),

        # ratios (annualized, RF)
        "sharpe_annualized": sharpe_annualized_excess(p, risk_free_rate_annual, trading_days),
        "sortino_annualized": sortino_annualized_excess(p, risk_free_rate_annual, trading_days),

        # risk
        "max_drawdown": max_drawdown_from_nav(nav),

        # hit rates
        # "win_rate_all_days": win_rate_all_days(p),
        "win_rate_trade_days": win_rate_trade_days(df, profit_col, signal_col=signal_col),

        # signal activity
        "trade_attempt": trade_attempt_rate(df, signal_col=signal_col),
        "exposure": exposure_rate(df[signal_col]) if signal_col in df.columns else np.nan,
        "signal_turnover_yearly": signal_turnover_yearly(df[signal_col], trading_days=trading_days) if signal_col in df.columns else np.nan,

        # nav
        "final_nav": float(nav.iloc[-1]) if len(nav) else np.nan,

        
    }

    # Stationarity / mean-reversion diagnostics (ADF p-value + half-life)
    metrics["stationarity"] = stationarity_diagnostics(
        df,
        price_col="price",
        zscore_window=4,
        min_obs=20,
    )

    return metrics, nav