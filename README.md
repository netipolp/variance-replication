last review. is there any typo or serious mistake

# Variance Trading via Portfolio Replication

Research and production implementation of variance trading through static option portfolio replication, following Derman & Miller’s framework.

This repository contains:

1. **Research version (Jupyter Notebook)** — exploratory analysis and signal development  
2. **Packaged Python framework (varrep)** — modular backtesting framework  

---

## Research Version (Notebook)

### Summary

This research explores variance trading through a replicating portfolio of vanilla options, following Derman & Miller’s framework. An implied variance measure is constructed from SPY option prices (2020–2022) and mean-reversion–based trading signals are applied, including composite Bollinger Bands and EMA filters. Multiple signal horizons and half-life adjustments are tested to evaluate profitability and signal robustness.

### Files

- `Variance_Replicating_Portfolio.ipynb` — full analysis, code, and results  
- `data/spy_2020_2022.csv` — input option dataset from Kaggle  

### Data Source

SPY daily end-of-day option quotes (2020–2022):  
https://www.kaggle.com/datasets/kylegraupe/spy-daily-eod-options-quotes-2020-2022  

### How to Run

Open `Variance_Replicating_Portfolio.ipynb` in Jupyter Notebook or Google Colab and run all cells sequentially.

---

## Package Framework (Varrep)

### Project Structure

```text
varrep/
├── src/
│   └── varrep/
│       ├── config.py          # Central configuration
│       ├── loader.py          # Data ingestion + cleaning
│       ├── replicator.py      # Variance replication logic
│       ├── backtest.py        # Entry/exit + portfolio evolution
│       └── metrics.py         # Performance & statistical metrics
├── scripts/
│   └── run_backtest.py        # CLI entry point
├── data/                      # Input CSV files
└── README.md
```

### Data Requirements

Input CSV must contain:

- `QUOTE_DATE`  
- `EXPIRE_DATE`  
- `STRIKE`  
- `UNDERLYING_LAST`  
- `C_BID`, `C_ASK`  
- `P_BID`, `P_ASK`  

---

### Run Backtest (CLI)

Example:

```bash
python scripts/run_backtest.py --csv_path data/spy_2020_2022.csv --expiry_date 2022-12-30
```
Additional CLI arguments are documented in `scripts/run_backtest.py`

### Outputs

- `results_<timestamp>.pkl` — full backtest results  
- `portfolio_metrics_<timestamp>.json` — portfolio performance summary  
- `nav_<timestamp>.csv` — daily NAV  
- `run_<timestamp>.log` — run log  

---

### Methodology

1. Implied variance is constructed via a model-free replication argument: a continuum of OTM puts (for strikes below the underlying price) and OTM calls (above the underlying price) is combined to replicate a log contract. After scaling by \( 2/T \), the portfolio’s initial fair value equals the underlying’s variance under the BSM framework (subject to certain assumptions). Daily implied variance is obtained by numerically integrating option prices across strikes according to this replication formula.

2. The resulting variance series is treated as mean-reverting. Trading signals are generated using EMA deviation and Bollinger Band thresholds, with optional half-life and volatility filters.

3. Positions are entered at signal trigger and repriced using forward-available quotes. Daily PnL is aggregated to form a return series.

4. Performance is evaluated via Sharpe ratio, drawdown, and mean-reversion diagnostics across multiple parameter configurations.

---

### References

Derman, E. & Miller, M. B. (2016).  
*The Volatility Smile: An Introduction for Students and Practitioners.* Wiley Finance.

---

### Disclaimer

For academic or educational use only.  
Not investment advice.  

Large language model (LLM) tools were used for coding support and workflow acceleration.