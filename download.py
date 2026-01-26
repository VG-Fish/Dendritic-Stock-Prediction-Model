# Code modified from https://www.kaggle.com/code/jacksoncrow/download-nasdaq-historical-data

import logging
import os
import shutil
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import redirect_stderr
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from pathlib import Path
from typing import Dict, List, Optional, Self, Tuple, Union

import pandas as pd
import polars as pl
import yfinance as yf
from tqdm import tqdm

from parse_config import ModelConfig

yf_logger: logging.Logger = logging.getLogger("yfinance")
yf_logger.setLevel(logging.CRITICAL)


@dataclass
class NASDAQDatasetInfo:
    parent_directory: Path
    stocks_directory: Path
    etfs_directory: Path
    valid_tickers_metadata: Path

    # Code modified from https://stackoverflow.com/a/57896232
    @classmethod
    def create_new(
        cls: type, parent_directory: Path, make_unique: bool = False
    ) -> "NASDAQDatasetInfo":
        if not make_unique:
            return cls._from_parent(parent_directory)

        candidate: Path = parent_directory
        counter: int = 1
        while candidate.exists():
            candidate = parent_directory.with_name(f"{parent_directory.name}_{counter}")
            counter += 1

        return cls._from_parent(candidate)

    @classmethod
    def _from_parent(cls: type, parent: Path) -> "NASDAQDatasetInfo":
        return cls(
            parent_directory=parent,
            stocks_directory=parent / "stocks",
            etfs_directory=parent / "etfs",
            valid_tickers_metadata=parent / "symbols_valid_meta.csv",
        )


class NASDAQDatasetCreationOptions(IntEnum):
    """
    Options to determine what `NASDAQDownloader().download_dataset()` should do when creating a new dataset. All the options
    will only be applied if an existing dataset directory exists.

    `IGNORE` creates a new directory with the same name as the existing dataset directory but adds a number to the end.

    `OVERWRITE` will reuse the same directory and add new files if they don't exist already. More than, `target` number of files
    may exist if you use this option.

    `REPLACE` will remove all the files in the existing directory and start from scratch.

    `REUSE` will make `download_dataset()` return early, causing the existing dataset to be used.

    `STOP` will raise an error if the existing dataset directory exists.
    """

    IGNORE = 0
    OVERWRITE = 1
    REPLACE = 2
    REUSE = 3
    STOP = 4


class SecurityType(StrEnum):
    STOCK = "N"
    ETF = "Y"
    ALL = "A"


class NASDAQDownloader:
    def __init__(self: Self, model_config: ModelConfig) -> None:
        self.data: pl.DataFrame = (
            pl.read_csv(
                "http://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
                separator="|",
            )
            .filter(pl.col("Test Issue") == "N")
            .sample(fraction=1, shuffle=True, seed=model_config.random_seed)
        )

        self._stock_id_map: pl.DataFrame = self.data.with_columns(
            self.data["Symbol"],
            self.data["Symbol"].cast(pl.Categorical).to_physical().alias("Stock ID"),
        ).select("Symbol", "Stock ID")

        self.symbol_data: pl.DataFrame = self.data.select("Symbol", "ETF").unique(
            subset=["Symbol"], keep="first"
        )

    def _process_symbol(
        self: Self, save_directory: Path, symbol: str, is_etf: bool
    ) -> Tuple[str, bool]:
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

        # Sanitize symbol for filename (e.g., "BRK/B" -> "BRK-B")
        safe_symbol = symbol.replace("/", "-").replace("^", "-")
        sub_directory = "etfs" if is_etf else "stocks"
        stock_data_path = save_directory / sub_directory / f"{safe_symbol}.csv"

        if stock_data_path.exists():
            return symbol, True

        stock_data_pd: Optional[pd.DataFrame] = None

        # Try downloading with the largest period possible
        for period in reversed(periods):
            try:
                stock_data_pd = yf.download(symbol, period=period, progress=False)
            except Exception:
                continue

            if stock_data_pd is None or stock_data_pd.empty:
                continue
            # We're good to break now as everything downloaded successfully
            break

        if stock_data_pd is None:
            return symbol, False

        # Cleanup DataFrame structure
        if isinstance(stock_data_pd.columns, pd.MultiIndex):
            stock_data_pd.columns = stock_data_pd.columns.get_level_values(0)

        stock_data_pd = stock_data_pd.reset_index()

        # Only accept CSVs with 6 columns including Date, some have 7 columns
        if len(stock_data_pd.columns) != 6 or stock_data_pd.empty:
            return symbol, False

        try:
            stock_data = pl.from_pandas(stock_data_pd, include_index=True)
            stock_data.write_csv(stock_data_path)
        except Exception:
            return symbol, False

        return symbol, True

    def download_dataset(
        self: Self,
        save_directory: Union[str, Path],
        dataset_creation_option: NASDAQDatasetCreationOptions = NASDAQDatasetCreationOptions.IGNORE,
        security_type: SecurityType = SecurityType.ALL,
        target: Optional[int] = None,
    ) -> NASDAQDatasetInfo:
        save_directory = Path(save_directory)
        dataset_info: NASDAQDatasetInfo = NASDAQDatasetInfo.create_new(
            save_directory,
        )

        if save_directory.exists():
            match dataset_creation_option:
                case NASDAQDatasetCreationOptions.IGNORE:
                    print("Creating existing dataset...")
                    dataset_info = NASDAQDatasetInfo.create_new(
                        save_directory, make_unique=True
                    )
                case NASDAQDatasetCreationOptions.REPLACE:
                    print("Removing existing dataset...")
                    shutil.rmtree(save_directory)
                case NASDAQDatasetCreationOptions.REUSE:
                    print("Reusing existing dataset...")
                    return dataset_info
                case NASDAQDatasetCreationOptions.STOP:
                    raise FileExistsError(f"'{save_directory}' already exists.")

        os.makedirs(dataset_info.stocks_directory, exist_ok=True)
        os.makedirs(dataset_info.etfs_directory, exist_ok=True)

        candidates: pl.DataFrame = self.symbol_data
        match security_type:
            case SecurityType.STOCK:
                candidates = candidates.filter(pl.col("ETF") == "N")
            case SecurityType.ETF:
                candidates = candidates.filter(pl.col("ETF") == "Y")

        print(f"Found {len(candidates)} candidate symbols.")

        def submit_new_task(idx: int) -> int:
            symbol, etf_flag = candidate_rows[idx]
            future = executor.submit(
                self._process_symbol,
                dataset_info.parent_directory,
                symbol,
                etf_flag == "Y",
            )
            pending_futures[future] = idx
            return idx + 1

        total_available = len(candidates)
        target_downloads = target if target is not None else total_available
        target_downloads = min(target_downloads, total_available)

        candidate_rows: List[Tuple] = candidates.rows()
        valid_symbols: List[str] = []

        # The first two statements in this triple with statement removes all logs to stderr
        with (
            open(os.devnull, "w") as devnull,
            redirect_stderr(devnull),
            ThreadPoolExecutor(max_workers=32) as executor,
        ):
            pending_futures: Dict = {}
            next_idx: int = 0
            success_count: int = 0

            while next_idx < target_downloads:
                next_idx = submit_new_task(next_idx)

            with tqdm(
                total=target_downloads,
                desc="Downloading Dataset Progress",
                unit="file",
                file=sys.stdout,
            ) as pbar:
                while pending_futures and success_count < target_downloads:
                    # Wait for at least one future to complete
                    # FIRST_COMPLETED ensures we process results as soon as they arrive
                    done, _ = wait(pending_futures.keys(), return_when=FIRST_COMPLETED)

                    for future in done:
                        _ = pending_futures.pop(future)

                        is_success: bool
                        symbol: str = ""
                        try:
                            symbol, is_success = future.result()
                        except Exception:
                            is_success = False

                        if is_success:
                            success_count += 1
                            valid_symbols.append(symbol)
                            pbar.update(1)
                        elif next_idx < total_available:
                            next_idx = submit_new_task(next_idx)

                # If we hit the target, cancel any remaining tasks
                for f in pending_futures:
                    f.cancel()

        print(
            f"Download complete. "
            f"Target: {target_downloads}, Success: {success_count}, "
            f"Valid Files on Disk: {len(valid_symbols)}"
        )

        self.data.filter(pl.col("Symbol").is_in(valid_symbols)).write_csv(
            dataset_info.valid_tickers_metadata
        )

        return dataset_info

    @property
    def stock_id_map(self: Self) -> pl.DataFrame:
        return self._stock_id_map
