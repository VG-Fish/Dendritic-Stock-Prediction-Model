# Code modified from https://www.kaggle.com/code/jacksoncrow/download-nasdaq-historical-data

import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stderr
from typing import Dict, List, Optional

import pandas as pd
import polars as pl
import yfinance as yf
from tqdm import tqdm

period: str = "max"
DATA_DIRECTORY: str = "stock_market_dataset"

if not os.path.exists(DATA_DIRECTORY):
    os.mkdir(DATA_DIRECTORY)
    os.mkdir(f"{DATA_DIRECTORY}/stocks")
    os.mkdir(f"{DATA_DIRECTORY}/etfs")

data: pl.DataFrame = pl.read_csv(
    "http://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt", separator="|"
)
cleaned_data: pl.DataFrame = data.filter(pl.col("Test Issue") == "N")
symbols: pl.Series = cleaned_data["Symbol"]

total: int = len(symbols)
is_valid: List[bool] = [False] * len(symbols)


def process_symbol(i: int) -> bool:
    symbol: str = symbols[i]
    periods: List[str] = [
        "1d",
        "5d",
        "1mo",
        "3mo",
        "6mo",
        "1y",
        "2y",
        "5y",
        "10y",
        "max",
    ]
    stock_data_pd: Optional[pd.DataFrame] = None
    # Start with longest periods, if getting all the data is successful, break the loop.
    for period in reversed(periods):
        try:
            stock_data_pd = yf.download(symbol, period=period)
        except Exception:
            continue

        if stock_data_pd is None or stock_data_pd.empty:
            continue
        break

    # Safety check
    if stock_data_pd is None or stock_data_pd.empty:
        return False

    stock_data: pl.DataFrame = pl.from_pandas(stock_data_pd)  # pyright: ignore[reportCallIssue, reportArgumentType]

    etf_flag = cleaned_data[i]["ETF"][0]
    match etf_flag:
        case "Y":
            stock_data.write_csv(f"{DATA_DIRECTORY}/etfs/{symbol}.csv")
        case "N":
            stock_data.write_csv(f"{DATA_DIRECTORY}/stocks/{symbol}.csv")

    return True


# The triple with statement removes all logs to stderr
with (
    open(os.devnull, "w") as devnull,
    redirect_stderr(devnull),
    ThreadPoolExecutor(max_workers=32) as ex,
):
    futures: Dict = {ex.submit(process_symbol, i): i for i in range(total)}
    for future in tqdm(as_completed(futures), total=total, file=sys.stdout):
        i: int = futures[future]
        try:
            is_valid[i] = future.result()
        except Exception:
            is_valid[i] = False

    num_downloaded: int = sum(is_valid)
    print(
        f"Total percentage of valid symbols downloaded: {(num_downloaded / total * 100) = :.3f}%"
    )

    valid_data: pl.DataFrame = cleaned_data.filter(is_valid)
    valid_data.write_csv(f"{DATA_DIRECTORY}/symbols_valid_meta.csv")

# Python threads need to be shutdown, and it takes a while
print("Cleaning up...")
