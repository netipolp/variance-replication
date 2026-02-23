from dataclasses import dataclass
from typing import Literal, Optional

PriceUsed = Literal["last", "bid", "ask", "mid", "last_adj"]

@dataclass(frozen=True)
class Config:
    # data
    csv_path: str = "data/spy_2020_2022.csv"
    expiry_date: str = "2022-12-30"
    output_dir: str = "outputs"

    # strategy scope
    quote_date_start: Optional[str] = None  
    quote_date_end: Optional[str] = None
    max_quote_dates: Optional[int] = None   # 10

    # replication
    price_used: PriceUsed = "mid"

    # signal (set these to your notebook values if different)
    ema_span: int = 10
    std_span: int = 10
    volatility_span: int = 5
    bb_switch: float = 1.5
    volatility_benchmark: float = 0.01

    # execution / pnl
    days_to_shift: int = 1
    trading_fee: float = 0.005
