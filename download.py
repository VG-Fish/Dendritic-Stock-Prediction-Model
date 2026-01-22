# Code modified from https://www.kaggle.com/code/jacksoncrow/download-nasdaq-historical-data

import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stderr
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Dict, List, Optional, Self

import pandas as pd
import polars as pl
import yfinance as yf
from tqdm import tqdm

yf_logger: logging.Logger = logging.getLogger("yfinance")
yf_logger.setLevel(logging.CRITICAL)


@dataclass
class NASDAQDatasetInfo:
    parent_directory: Path
    stocks_directory: Path
    etfs_directory: Path
    valid_tickers_metadata: Path


class SecurityType(StrEnum):
    STOCK = "N"
    ETF = "Y"
    ALL = "A"


class NASDAQDownloader:
    def __init__(self: Self, data_directory: str = "nasdaq_dataset") -> None:
        self.data_directory: Path = Path(data_directory)
        self._dataset_info: NASDAQDatasetInfo = NASDAQDatasetInfo(
            self.data_directory,
            self.data_directory / "stocks",
            self.data_directory / "etfs",
            self.data_directory / "symbols_valid_meta.csv",
        )

        data: pl.DataFrame = pl.read_csv(
            "http://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt", separator="|"
        )
        self.cleaned_data: pl.DataFrame = data.filter(pl.col("Test Issue") == "N")
        self.symbol_data: pl.DataFrame = self.cleaned_data.select("Symbol", "ETF")

    def _process_symbol(self: Self, i: int) -> bool:
        symbol: str = self.symbol_data["Symbol"][i]
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
        if stock_data_pd is None:
            return False

        # Removes ticker name from MultiIndex columns
        if isinstance(stock_data_pd.columns, pd.MultiIndex):
            stock_data_pd.columns = stock_data_pd.columns.get_level_values(0)

        # This line ensures that the "Date" index gets saved as the "Date" index is turned into a column
        stock_data_pd = stock_data_pd.reset_index()

        # 6 is the number of columns we want, some CSVs have 7 columns with "Adjusted Close".
        # Adding those CSVs makes life more inconvenient, so I'm just ignoring them here.
        if len(stock_data_pd.columns) != 6 or stock_data_pd.empty:
            return False

        stock_data = pl.from_pandas(stock_data_pd, include_index=True)

        etf_flag = self.cleaned_data[i]["ETF"][0]
        stock_data_path: str = f"{self.data_directory}/{'etfs' if etf_flag == 'Y' else 'stocks'}/{symbol}.csv"
        if not os.path.exists(stock_data_path):
            stock_data.write_csv(stock_data_path)

        return True

    def download_dataset(
        self: Self,
        security_type: SecurityType,
        stop_if_dest_dir_exists: bool = True,
        target: Optional[int] = None,
    ) -> NASDAQDatasetInfo:
        if stop_if_dest_dir_exists and os.path.exists(self.data_directory):
            return self._dataset_info

        os.makedirs(self._dataset_info.stocks_directory, exist_ok=True)
        os.makedirs(self._dataset_info.etfs_directory, exist_ok=True)

        match security_type:
            case SecurityType.STOCK:
                self.symbol_data = self.symbol_data.filter(
                    pl.col("ETF") == SecurityType.STOCK
                )
            case SecurityType.ETF:
                self.symbol_data = self.symbol_data.filter(
                    pl.col("ETF") == SecurityType.ETF
                )
            case SecurityType.ALL:
                pass
        print(self.symbol_data)

        num_symbols: int = target or len(self.symbol_data)
        is_valid: List[bool] = [False] * len(self.symbol_data)

        # The first two statements in this triple with statement removes all logs to stderr
        with (
            open(os.devnull, "w") as devnull,
            redirect_stderr(devnull),
            ThreadPoolExecutor(max_workers=32) as ex,
        ):
            futures: Dict = {
                ex.submit(self._process_symbol, i): i for i in range(num_symbols)
            }
            for future in tqdm(
                as_completed(futures),
                total=num_symbols,
                file=sys.stdout,
                desc="Downloading Dataset...",
            ):
                i: int = futures[future]
                try:
                    is_valid[i] = future.result()
                except Exception:
                    is_valid[i] = False

            num_downloaded: int = sum(is_valid)
            print(
                f"Total percentage of valid symbols downloaded: {(num_downloaded / num_symbols * 100) = :.3f}%"
            )

        self.symbol_data.filter(is_valid).write_csv(
            self._dataset_info.valid_tickers_metadata
        )

        # Python threads need to be shutdown, and it takes a while
        print("Cleaning up...")
        return self._dataset_info
