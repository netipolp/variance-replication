from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

PriceUsed = Literal["last", "bid", "ask", "mid", "last_adj"]


@dataclass
class ReplicationResult:
    vol: float
    variance: float
    n_options: int
    rep_port: pd.DataFrame


class VarianceReplicator:

    def __init__(self, price_used: PriceUsed = "mid"):
        self.price_used = price_used

    def build_rep_port(self, options_data: pd.DataFrame, quote_date: str, expiry_date: str) -> ReplicationResult:
        """
        Notebook-equivalent of IV_rep_port(options_data, t, T, price_used).
        Returns vol, variance, length, combined_data (rep_port).
        """
        filtered = options_data[
            (options_data["t"].astype(str).str.strip() == quote_date)
            & (options_data["T"].astype(str).str.strip() == expiry_date)
        ].copy()

        filtered = filtered.sort_values(by="K", ascending=True).copy()
        filtered = filtered[filtered["DTE"] > 0].copy()

        if filtered.empty:
            return ReplicationResult(vol=np.nan, variance=np.nan, n_options=0, rep_port=pd.DataFrame())

        _1_forward = (filtered["K"] - filtered["S"]) / filtered["S"]
        _2_log_contract = np.log(filtered["K"] / filtered["S"])
        filtered["pi"] = (2 * 365 / filtered["DTE"]) * (_1_forward - _2_log_contract)

        # puts
        put = filtered[filtered["K"] < filtered["S"]].copy()
        put = put[["t", "T", "S", "K", "pi", "P_LAST", "P_BID", "P_ASK"]]
        put["side"] = "P"
        put["lambda"] = ((put["pi"].diff()) / (put["K"].diff())).abs()
        put["w"] = put["lambda"] - put["lambda"].shift(-1)
        if not put.empty:
            put.at[put.index[-1], "w"] = put.at[put.index[-1], "lambda"]

        # calls
        call = filtered[filtered["K"] >= filtered["S"]].copy()
        call = call[["t", "T", "S", "K", "pi", "C_LAST", "C_BID", "C_ASK"]]
        call["side"] = "C"
        call["lambda"] = (-call["pi"].diff(-1) / call["K"].diff(-1)).abs()
        call["w"] = call["lambda"] - call["lambda"].shift(1)
        if not call.empty:
            call.at[call.index[0], "w"] = call.at[call.index[0], "lambda"]

        combined = pd.concat([put, call], ignore_index=True)
        if combined.empty:
            return ReplicationResult(vol=np.nan, variance=np.nan, n_options=0, rep_port=pd.DataFrame())

        center_idx = (combined["K"] - combined["S"]).abs().idxmin()
        combined["center"] = 0
        combined.loc[center_idx, "center"] = 1

        combined = combined.dropna(subset=["w"]).copy()

        # numeric prices + mid
        for c in ["P_LAST", "C_LAST", "P_ASK", "C_ASK", "P_BID", "C_BID"]:
            if c in combined.columns:
                combined[c] = pd.to_numeric(combined[c], errors="coerce")

        combined["P_MID"] = (combined.get("P_BID") + combined.get("P_ASK")) / 2
        combined["C_MID"] = (combined.get("C_BID") + combined.get("C_ASK")) / 2

        price_map = {
            "last": lambda row: row["P_LAST"] if row["side"] == "P" else row["C_LAST"],
            "bid":  lambda row: row["P_BID"]  if row["side"] == "P" else row["C_BID"],
            "ask":  lambda row: row["P_ASK"]  if row["side"] == "P" else row["C_ASK"],
            "mid":  lambda row: row["P_MID"]  if row["side"] == "P" else row["C_MID"],
            "last_adj": lambda row: (
                row["P_LAST"] if (row["side"] == "P" and pd.notna(row["P_LAST"]) and row["P_LAST"] != 0)
                else row["P_MID"] if row["side"] == "P"
                else row["C_LAST"] if (pd.notna(row["C_LAST"]) and row["C_LAST"] != 0)
                else row["C_MID"]
            ),
        }
        if self.price_used not in price_map:
            raise ValueError(f"Invalid price_used={self.price_used}. Use one of {list(price_map.keys())}")

        combined["w_O"] = combined["w"] * combined.apply(price_map[self.price_used], axis=1)

        variance = float(combined["w_O"].sum())
        vol = float(np.sqrt(variance)) if np.isfinite(variance) and variance >= 0 else np.nan

        return ReplicationResult(vol=vol, variance=variance, n_options=len(combined), rep_port=combined)
    def reprice_rep_port_exit(
        self,
        rep_port_entry: pd.DataFrame,
        options_data: pd.DataFrame,
        days_to_shift: int = 1,
    ) -> pd.DataFrame:
        """
        Reprice replication portfolio at exit date.

        - Normalizes dates robustly
        - Prevents infinite forward-date loop
        - Returns empty DataFrame if no valid exit date found
        """

        if rep_port_entry is None or rep_port_entry.empty:
            return pd.DataFrame()

        if "t" not in rep_port_entry.columns:
            return pd.DataFrame()

        rep = rep_port_entry.copy()


        # --- Find next valid date ---
        rep['t'] = pd.to_datetime(rep['t'])
        t0 = rep["t"].iloc[0]
        next_date = t0 + pd.Timedelta(days=days_to_shift)
            
        # while next_date.strftime('%Y-%m-%d') not in options_data['t'].unique():
        #     next_date += pd.Timedelta(days=1)  # Keep shifting forward until a valid date is found
    

        max_forward_days = 30
        steps = 0

        while next_date.strftime('%Y-%m-%d') not in options_data['t'].unique():
            next_date += pd.Timedelta(days=1)  # Keep shifting forward until a valid date is found
            steps += 1
            if steps > max_forward_days:
                return pd.DataFrame()

        exit_date_str = next_date.strftime("%Y-%m-%d")

        # --- Update t ---
        rep["t"] = exit_date_str

        # right before merge in reprice_rep_port_exit()

        rep["t"] = pd.to_datetime(rep["t"], errors="coerce").dt.strftime("%Y-%m-%d")
        rep["T"] = pd.to_datetime(rep["T"], errors="coerce").dt.strftime("%Y-%m-%d")
        rep["K"] = pd.to_numeric(rep["K"], errors="coerce").round(8)

        # --- Merge new prices ---
        rep = rep.merge(
            options_data[["t", "T", "K", "P_MID", "C_MID"]],
            on=["t", "T", "K"],
            how="left",
            suffixes=("", "_new"),
        )

        rep["P_MID_new"] = pd.to_numeric(rep["P_MID_new"], errors="coerce")
        rep["C_MID_new"] = pd.to_numeric(rep["C_MID_new"], errors="coerce")

        # --- Recompute option value ---
        rep["w_O"] = rep["w"] * rep.apply(
            lambda row: row["P_MID_new"] if row["side"] == "P" else row["C_MID_new"],
            axis=1,
        )

        return rep
