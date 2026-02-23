import logging
import time

import pandas as pd

from varrep.variance_replication import VarianceReplicator
from varrep.signals import compute_trading_signals

logger = logging.getLogger(__name__)


def _progress_log(i: int, n: int, t0: float, every: int = 50, prefix: str = "Progress"):
    """
    Lightweight progress logger (INFO) with ETA.
    No impact on logic; safe to call inside loops.
    """
    if n <= 0:
        return
    if i == 1 or i == n or (every > 0 and i % every == 0):
        elapsed = time.time() - t0
        rate = i / elapsed if elapsed > 0 else float("inf")
        eta = (n - i) / rate if rate > 0 else float("inf")
        logger.info("%s: %d/%d (%.1f%%) | %.2f it/s | ETA %.1fs", prefix, i, n, 100 * i / n, rate, eta)


def build_results_for_expiry(options_data: pd.DataFrame, expiry_date: str, price_used: str = "mid") -> pd.DataFrame:
    """
    Builds term-structure table for ONE expiry.
    Mirrors your notebook loop over unique_pairs_used.
    """
    t_all = time.time()
    logger.info("build_results_for_expiry: start | expiry=%s | rows=%d", expiry_date, len(options_data))

    unique_pairs = options_data[["t", "T", "DTE"]].drop_duplicates()
    unique_pairs = unique_pairs[unique_pairs["t"] != unique_pairs["T"]]  # delete intraday quote
    unique_pairs_used = unique_pairs[unique_pairs["T"].astype(str) == expiry_date].copy()

    logger.info("build_results_for_expiry: unique_pairs_used=%d", len(unique_pairs_used))

    replicator = VarianceReplicator(price_used=price_used)

    results = []
    n = len(unique_pairs_used)
    t0 = time.time()

    for i, (_, row) in enumerate(unique_pairs_used.iterrows(), 1):
        _progress_log(i, n, t0, every=50, prefix="build_results_for_expiry")

        t = str(row["t"])
        T = str(row["T"])
        dte = float(row["DTE"])

        try:
            res = replicator.build_rep_port(options_data, t, T)
            results.append(
                {
                    "t": t,
                    "T": T,
                    "DTE": dte,
                    "vol_mid": res.vol,
                    "variance_mid": res.variance,
                    "data_length_mid": res.n_options,
                    "rep_port_mid": res.rep_port,
                }
            )
        except Exception:
            logger.exception("build_rep_port failed | t=%s | T=%s | DTE=%s", t, T, dte)
            results.append(
                {
                    "t": t,
                    "T": T,
                    "DTE": dte,
                    "vol_mid": None,
                    "variance_mid": None,
                    "data_length_mid": 0,
                    "rep_port_mid": None,
                }
            )

    df = pd.DataFrame(results).reset_index(drop=True)
    df = df.sort_values(by="DTE", ascending=False).reset_index(drop=True)

    # notebook uses results_df['price'] for signal computation (price == vol_mid)
    df["price"] = df["vol_mid"]

    logger.info("build_results_for_expiry: done | out_rows=%d | elapsed=%.2fs", len(df), time.time() - t_all)
    return df


def run_backtest_for_expiry(
    options_data: pd.DataFrame,
    expiry_date: str,
    price_used: str = "mid",
    ema_span: int = 10,
    std_span: int = 10,
    volatility_span: int = 5,
    bb_switch: float = 1.5,
    volatility_benchmark: float = 0.01,
    days_to_shift: int = 1,
    trading_fee: float = 0.005,
) -> pd.DataFrame:
    """
    End-to-end for one expiry:
    1) build term structure
    2) signals
    3) entry shift (t+1), exit repricing (t+1 shifted by days_to_shift)
    4) pnl columns with notebook names
    """
    t_all = time.time()
    logger.info("Backtest: start | expiry=%s | rows=%d", expiry_date, len(options_data))

    logger.info("Backtest: step 1/4 build term structure")
    results_df = build_results_for_expiry(options_data, expiry_date=expiry_date, price_used=price_used)
    logger.info("Backtest: term structure built | rows=%d", len(results_df))

    logger.info("Backtest: step 2/4 compute signals")
    results_df = compute_trading_signals(
        results_df,
        ema_span=ema_span,
        std_span=std_span,
        volatility_span=volatility_span,
        bb_switch=bb_switch,
        volatility_benchmark=volatility_benchmark,
    )
    logger.info("Backtest: signals done")

    results_df = results_df.reset_index(drop=True)

    logger.info("Backtest: step 3/4 entry shift + repricing exits (days_to_shift=%d)", days_to_shift)

    # entry shift (like notebook)
    results_df["t_entry"] = results_df["t"].shift(-1)
    results_df["variance_entry"] = results_df["variance_mid"].shift(-1)
    results_df["rep_port_entry"] = results_df["rep_port_mid"].shift(-1)
    results_df["no_O_entry"] = results_df["rep_port_entry"].apply(lambda df: len(df) if isinstance(df, pd.DataFrame) else 0)

    # exit repricing (keep weights, reprice on next available date)
    replicator = VarianceReplicator(price_used=price_used)

    results_df["rep_port_exit"] = None
    max_valid_index = len(results_df) - days_to_shift

    t0 = time.time()
    for i, idx in enumerate(range(max_valid_index), 1):
        _progress_log(i, max_valid_index, t0, every=50, prefix="reprice_rep_port_exit")

        rep_entry = results_df.loc[idx, "rep_port_entry"]
        if not isinstance(rep_entry, pd.DataFrame) or rep_entry.empty:
            continue
        rep_exit = replicator.reprice_rep_port_exit(rep_entry, options_data, days_to_shift=days_to_shift)
        results_df.at[idx, "rep_port_exit"] = rep_exit

    logger.info("Backtest: repricing done")

    results_df["t_exit"] = results_df["rep_port_exit"].apply(
        lambda df: df["t"].iloc[0] if isinstance(df, pd.DataFrame) and not df.empty else None
    )
    results_df["variance_exit"] = results_df["rep_port_exit"].apply(
        lambda df: float(df["w_O"].sum()) if isinstance(df, pd.DataFrame) and not df.empty else None
    )
    results_df["no_O_exit"] = results_df["rep_port_exit"].apply(lambda df: len(df) if isinstance(df, pd.DataFrame) else 0)

    logger.info("Backtest: step 4/4 compute pnl columns")

    # pct change + notebook-named profit columns
    results_df["pct_change"] = results_df["variance_exit"] / results_df["variance_entry"] * (1 - trading_fee) - 1
    results_df["profit_entry@t1_exit@t2"] = results_df["pct_change"] * results_df["signal2"]
    results_df["profit_entry@t0_exit@t1"] = results_df["pct_change"].shift(1) * results_df["signal2"]

    logger.info("Backtest: done | elapsed=%.2fs", time.time() - t_all)
    return results_df
