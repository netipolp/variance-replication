# varrep

Variance-replicating portfolio backtest packaged as a runnable Python project.

## Install (editable)
```bash
pip install -e .
```

## Run
```bash
python scripts/run_backtest.py --csv data/spy_2020_2022.csv --expiry 2021-03-31
```

Outputs:
- `results_<timestamp>.pkl` (backtest table)
- console metrics summary (logging)
