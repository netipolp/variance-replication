#!/usr/bin/env python3

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
import json

from varrep.config import Config
from varrep.loader import load_options_csv
from varrep.backtest import run_backtest_for_expiry
from varrep.metrics import full_portfolio_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run variance replication backtest + portfolio metrics")

    # align CLI names with Config field names (override-only; None => use config default)
    p.add_argument("--csv_path", dest="csv_path", default=None, help="CSV path (override)")
    p.add_argument("--expiry_date", dest="expiry_date", default=None, help="Expiry date YYYY-MM-DD (override)")
    p.add_argument("--price_used", dest="price_used", default=None, choices=["mid"], help="Pricing mode (override)")
    p.add_argument("--output_dir", dest="output_dir", default=None, help="Output directory (override)")

    # fast-run filter knobs (override)
    p.add_argument("--quote_date_start", dest="quote_date_start", default=None, help="Quote start YYYY-MM-DD (override)")
    p.add_argument("--quote_date_end", dest="quote_date_end", default=None, help="Quote end YYYY-MM-DD (override)")
    p.add_argument("--max_quote_dates", dest="max_quote_dates", type=int, default=None, help="Keep first N quote dates (override)")

    # strategy knobs (override)
    p.add_argument("--ema_span", dest="ema_span", type=int, default=None)
    p.add_argument("--std_span", dest="std_span", type=int, default=None)
    p.add_argument("--volatility_span", dest="volatility_span", type=int, default=None)
    p.add_argument("--bb_switch", dest="bb_switch", type=float, default=None)
    p.add_argument("--volatility_benchmark", dest="volatility_benchmark", type=float, default=None)
    p.add_argument("--days_to_shift", dest="days_to_shift", type=int, default=None)
    p.add_argument("--trading_fee", dest="trading_fee", type=float, default=None)

    # performance metric knobs (override)
    p.add_argument("--profit_col", dest="profit_col", default=None, help="Profit column name (override)")
    p.add_argument("--risk_free_rate_annual", dest="risk_free_rate_annual", type=float, default=None)
    p.add_argument("--trading_days", dest="trading_days", type=int, default=None)

    # logging
    p.add_argument("--log_to_file", dest="log_to_file", action="store_true", help="Also write logs to a .log file in output_dir")

    return p.parse_args()


def pick(cli_value, cfg_value):
    return cfg_value if cli_value is None else cli_value


def main():
    args = parse_args()
    cfg = Config()

    # resolve outputs early (so we can place log file there)
    output_dir = Path(pick(args.output_dir, getattr(cfg, "output_dir", "outputs")))
    output_dir.mkdir(parents=True, exist_ok=True)

    # logging: console always; optional file
    handlers = [logging.StreamHandler(sys.stdout)]
    if args.log_to_file:
        log_path = output_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        handlers.append(logging.FileHandler(log_path))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )
    logger = logging.getLogger(__name__)

    t0 = time.time()
    logger.info("Run started")

    # resolve config-with-overrides
    csv_path = Path(pick(args.csv_path, cfg.csv_path))

    # IMPORTANT: your Config uses expiry_date (per your preference)
    expiry_date = pick(args.expiry_date, cfg.expiry_date)

    price_used = pick(args.price_used, getattr(cfg, "price_used", "mid"))

    quote_date_start = pick(args.quote_date_start, getattr(cfg, "quote_date_start", None))
    quote_date_end = pick(args.quote_date_end, getattr(cfg, "quote_date_end", None))
    max_quote_dates = pick(args.max_quote_dates, getattr(cfg, "max_quote_dates", None))

    ema_span = pick(args.ema_span, cfg.ema_span)
    std_span = pick(args.std_span, cfg.std_span)
    volatility_span = pick(args.volatility_span, cfg.volatility_span)
    bb_switch = pick(args.bb_switch, cfg.bb_switch)
    volatility_benchmark = pick(args.volatility_benchmark, cfg.volatility_benchmark)
    days_to_shift = pick(args.days_to_shift, cfg.days_to_shift)
    trading_fee = pick(args.trading_fee, cfg.trading_fee)

    profit_col = pick(args.profit_col, "profit_entry@t1_exit@t2")
    risk_free_rate_annual = pick(args.risk_free_rate_annual, 0.04)
    trading_days = pick(args.trading_days, 252)

    logger.info("Loading CSV: %s", csv_path)
    options = load_options_csv(csv_path)
    logger.info("CSV loaded: %d rows", len(options))

    # -----------------------
    # FAST RUN FILTER
    # -----------------------
    if quote_date_start:
        before = len(options)
        options = options[options["t"] >= quote_date_start]
        logger.info("Filter quote_date_start=%s | %d -> %d rows", quote_date_start, before, len(options))

    if quote_date_end:
        before = len(options)
        options = options[options["t"] <= quote_date_end]
        logger.info("Filter quote_date_end=%s | %d -> %d rows", quote_date_end, before, len(options))

    if max_quote_dates:
        keep = sorted(options["t"].unique())[: max_quote_dates]
        before = len(options)
        options = options[options["t"].isin(keep)]
        logger.info("Filter max_quote_dates=%s | %d -> %d rows", max_quote_dates, before, len(options))

    # -----------------------
    # BACKTEST
    # -----------------------
    logger.info("Running backtest for expiry_date=%s ...", expiry_date)
    results = run_backtest_for_expiry(
        options_data=options,
        expiry_date=expiry_date,
        price_used=price_used,
        ema_span=ema_span,
        std_span=std_span,
        volatility_span=volatility_span,
        bb_switch=bb_switch,
        volatility_benchmark=volatility_benchmark,
        days_to_shift=days_to_shift,
        trading_fee=trading_fee,
    )
    logger.info("Backtest complete. Rows: %d", len(results))

    # -----------------------
    # SAVE PICKLE
    # -----------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"results_{timestamp}.pkl"
    results.to_pickle(output_path)
    logger.info("Saved: %s", output_path)

    # -----------------------
    # PORTFOLIO METRICS
    # -----------------------
    metrics, nav = full_portfolio_metrics(results, profit_col=profit_col)

    logger.info("--- Portfolio Metrics (%s) ---", profit_col)

    # Log non-nested metrics first
    for k in metrics.keys():
        if k != "stationarity":
            logger.info("%s: %s", k, metrics[k])

    # ---- Stationarity block ----
    diag = metrics.get("stationarity", {})

    logger.info("--- Stationarity Diagnostics (ADF + Half-life) ---")

    for name in ["raw", "diff", "zscore"]:
        d = diag.get(name, {})
        logger.info(
            "%s | n=%s | adf_stat=%s | p_value=%s | half_life=%s",
            name,
            d.get("n_obs"),
            d.get("adf_stat"),
            d.get("p_value"),
            d.get("half_life"),
        )

    if diag.get("note"):
        logger.warning("Stationarity note: %s", diag["note"])


    # -----------------------   
    # SAVE PORTFOLIO OUTPUTS
    # -----------------------
    # 1) metrics dict -> JSON
    metrics_path = output_dir / f"portfolio_metrics_{timestamp}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2, default=str)
    logger.info("Saved portfolio metrics: %s", metrics_path)

    # 2) nav series/dataframe -> CSV
    # nav might be Series or DataFrame depending on your metrics.py
    nav_path = output_dir / f"nav_{timestamp}.csv"
    try:
        nav.to_csv(nav_path, index=True)
    except Exception:
        # fallback: convert to DataFrame then save
        import pandas as pd
        pd.DataFrame(nav).to_csv(nav_path, index=True)
    logger.info("Saved NAV: %s", nav_path)

    logger.info("Done in %.2fs", time.time() - t0)


if __name__ == "__main__":
    main()


'''
PYTHONPATH="$(pwd)/src" python scripts/run_backtest.py \ 
  --csv_path data/spy_2020_2022.csv \
  --expiry_date 2022-12-30 \
  --log_to_file
'''