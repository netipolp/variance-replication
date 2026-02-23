import pandas as pd

REQUIRED_COLUMNS = {
    "UNDERLYING_LAST", "STRIKE", "QUOTE_DATE", "EXPIRE_DATE", "DTE",
    "P_ASK", "P_BID", "C_ASK", "C_BID",
    "P_LAST", "C_LAST",
}

RENAME_MAP = {
    "UNDERLYING_LAST": "S",
    "STRIKE": "K",
    "QUOTE_DATE": "t",
    "EXPIRE_DATE": "T",
}


def _require_columns(df: pd.DataFrame, cols: set[str]) -> None:
    missing = sorted(list(cols - set(df.columns)))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def load_options_csv(path: str) -> pd.DataFrame:
    """
    Loads your options CSV and standardizes columns to match notebook logic:
    - Rename: UNDERLYING_LAST->S, STRIKE->K, QUOTE_DATE->t, EXPIRE_DATE->T
    - Create: P_MID, C_MID
    - Numeric coercion for pricing columns
    """
    df = pd.read_csv(path, low_memory=False)


    # clean weird bracket chars if any (notebook did similar)
    df.rename(columns=lambda x: str(x).replace("[", "").replace("]", "").strip(), inplace=True)

    _require_columns(df, REQUIRED_COLUMNS)

    # keep dates as strings (like notebook comparisons)
    df["EXPIRE_DATE"] = df["EXPIRE_DATE"].str.strip()
    df["QUOTE_DATE"] = df["QUOTE_DATE"].str.strip()

    df = df.rename(columns=RENAME_MAP)

    # numeric_cols = ["P_ASK", "P_BID", "C_ASK", "C_BID", "P_LAST", "C_LAST", "S", "K", "DTE"]
    numeric_cols = ["P_ASK", "P_BID", "C_ASK", "C_BID", "P_LAST", "C_LAST"]
    # for c in numeric_cols:
    #     df[c] = pd.to_numeric(df[c], errors="coerce")
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    df["P_MID"] = (df["P_BID"] + df["P_ASK"]) / 2
    df["C_MID"] = (df["C_BID"] + df["C_ASK"]) / 2

    # df = df.dropna(subset=["t", "T", "S", "K", "DTE"]).copy()
    # df = df[df["DTE"] > 0].copy()

    # df["t"] = df["t"].astype(str).str.strip()
    # df["T"] = df["T"].astype(str).str.strip()

    # return df

    # df["t"] = pd.to_datetime(df["t"], errors="coerce").dt.strftime("%Y-%m-%d")
    # df["T"] = pd.to_datetime(df["T"], errors="coerce").dt.strftime("%Y-%m-%d")
    # df["K"] = pd.to_numeric(df["K"], errors="coerce").round(8)

    # df = df.dropna(subset=["t", "T", "K"]).copy()

    # optional: keep only what you need for repricing merges
    return df
