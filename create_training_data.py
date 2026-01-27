import shutil
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Dict, List, Self, Set

import numpy as np
import polars as pl
from numpy.random import Generator
from torch.utils.data.dataloader import DataLoader

from parse_config import ModelConfig
from stocks import NASDAQDataLoaders, NASDAQDataset

model_config: ModelConfig
# ANSI escape codes
RED: str = "\033[31m"
RESET: str = "\033[0m"


@dataclass
class SplitDFDatasets:
    train: pl.DataFrame
    val: pl.DataFrame
    test: pl.DataFrame


class DatasetLoadingConfig(IntEnum):
    """
    Options to determine what `TrainingDataCreator().create_data_loaders_from()` should do when creating a new dataset. All the options
    will only be applied if an existing `scaled_dataset` directory exists.

    `IGNORE` creates a new directory with the same name as the existing dataset directory but adds a number to the end.

    `REPLACE` will remove all the files in the existing directory and start from scratch.

    `REUSE` will make reuse the existing dataset.

    `STOP` will raise an error if the existing dataset directory exists.
    """

    IGNORE = 1
    REPLACE = 2
    REUSE = 3
    STOP = 4

    # Code modified from https://stackoverflow.com/a/57896232
    @classmethod
    def create_unique_path_from(
        cls: type,
        parent_directory: Path,
    ) -> Path:
        candidate: Path = parent_directory
        counter: int = 1
        while candidate.exists():
            candidate = parent_directory.with_name(f"{parent_directory.name}_{counter}")
            counter += 1

        return candidate


class TrainingDataCreator:
    def __init__(
        self: Self, model_config: ModelConfig, stock_id_map: pl.DataFrame
    ) -> None:
        self.model_config: ModelConfig = model_config
        self.stock_id_map: pl.DataFrame = stock_id_map

    # Gemini suggestion to improve dimensional accuracy
    # RSI calculates if the stock was overbought
    def _compute_rsi(self: Self, expr: pl.Expr, period: int = 14) -> pl.Expr:
        delta: pl.Expr = expr.diff()
        up: pl.Expr = delta.clip(lower_bound=0)
        down: pl.Expr = -delta.clip(upper_bound=0)
        ma_up: pl.Expr = up.ewm_mean(com=period - 1, ignore_nulls=True)
        ma_down: pl.Expr = down.ewm_mean(com=period - 1, ignore_nulls=True)
        rs: pl.Expr = ma_up / ma_down

        rsi: pl.Expr = 100 - (100 / (1 + rs))
        # Make NaNs neutral (50)
        return rsi.fill_nan(50)

    def _create_datasets_from(self: Self, directory: Path) -> SplitDFDatasets:
        def sanitize(symbol: str) -> str:
            return symbol.replace("/", "-").replace("^", "-")

        print(f"Scanning for CSVs in {directory}...")

        glob: List = list(directory.glob("*.csv"))
        print(f"Going through {len(glob)} CSVs...")
        schema_overrides: Dict[str, type] = {
            "Date": pl.Datetime,
            "Open": pl.Float64,
            "High": pl.Float64,
            "Low": pl.Float64,
            "Close": pl.Float64,
            "Volume": pl.Float64,
        }

        self.stock_id_map = self.stock_id_map.with_columns(
            pl.col("Symbol")
            .map_elements(
                lambda s: str(directory / f"{sanitize(s)}.csv"),
                return_dtype=pl.String,
            )
            .alias("Path")
        ).select("Path", "Stock ID")

        df: pl.DataFrame = (
            (
                pl.scan_csv(
                    glob,
                    include_file_paths="Path",
                    try_parse_dates=True,
                    schema_overrides=schema_overrides,
                )
                .sort(["Path", "Date"])
                .with_columns(
                    pl.col("Date").dt.date(),
                    pl.int_range(pl.len()).over("Path").alias("Index"),
                    pl.col("Close").log().diff().over("Path").alias("Log Return"),
                    (
                        pl.col("Open").log()
                        - pl.col("Close").log().shift(1).over("Path")
                    ).alias("Log Overnight"),
                    (pl.col("High").log() - pl.col("Low").log()).alias("Log Range"),
                    pl.col("Volume")
                    .log1p()
                    .diff()
                    .over("Path")
                    .alias("Log Volume Change"),
                )
                .with_columns(
                    self._compute_rsi(pl.col("Close"), period=14)
                    .over("Path")
                    .alias("RSI"),
                    (
                        pl.col("Close").ewm_mean(span=12)
                        - pl.col("Close").ewm_mean(span=26)
                    )
                    .over("Path")
                    .alias("MACD"),  # MACD tells the LSTM the strength of the trend.
                    # This is a trend feature that allows the model to determine if it's currently a bull or bear market
                    (pl.col("Close") / pl.col("Close").rolling_mean(window_size=50))
                    .log()
                    .over("Path")
                    .alias("SMA Ratio"),
                )
                # Calculate MACD Signal line (9 EMA of MACD)
                # EMA = exponential moving average, which places more weight on more recent data points
                .with_columns(
                    pl.col("MACD").ewm_mean(span=9).over("Path").alias("MACD Signal"),
                    pl.col("Log Return")
                    .rolling_std(window_size=self.model_config.sequence_length)
                    .over("Path")
                    .alias("Rolling STD"),
                    # Shift Log Return to the left to see tomorrow's return today
                    pl.col("Log Return").shift(-1).over("Path").alias("Target"),
                )
                .with_columns(
                    pl.all().exclude("Path", "Date", "Index").cast(pl.Float32)
                )
                .drop("High", "Low", "Close", "Volume", "Open")
                # Remove NaNs, nulls, and infinities
                .fill_nan(None)
                .filter(
                    pl.all_horizontal(
                        pl.all().exclude("Path", "Index", "Date", "Target").is_finite()
                    ),
                )
            )
            .collect()
            # This with statement normalizes the data for each feature column
            .with_columns(
                [
                    (
                        (
                            pl.col(c)
                            - pl.col(c).rolling_mean(window_size=60).over("Path")
                        )
                        / (pl.col(c).rolling_std(window_size=60).over("Path") + 1e-8)
                    ).alias(c)
                    for c in [
                        "Log Return",
                        "Log Overnight",
                        "Log Volume Change",
                        "Log Range",
                        "RSI",
                        "MACD",
                        "MACD Signal",
                        "SMA Ratio",
                        "Rolling STD",
                    ]
                ]
            )
            .drop_nulls()
            .join(
                self.stock_id_map,
                on="Path",
                how="inner",
            )
            .sort(["Path", "Index"])
            .rolling(
                index_column="Index",
                period=f"{self.model_config.sequence_length}i",
                group_by="Path",
            )
            .having(pl.len() == self.model_config.sequence_length)
            .agg(
                pl.col("Target").last(),
                pl.col("Date").last(),
                pl.col("Stock ID").last(),
                pl.col("RSI"),
                pl.col("MACD"),
                pl.col("MACD Signal"),
                pl.col("SMA Ratio"),
                pl.col("Rolling STD"),
                pl.col("Log Return")
                .slice(0, self.model_config.sequence_length)
                .alias("Close"),
                pl.col("Log Overnight")
                .slice(0, self.model_config.sequence_length)
                .alias("Open"),
                pl.col("Log Volume Change")
                .slice(0, self.model_config.sequence_length)
                .alias("Volume"),
                pl.col("Log Range")
                .slice(0, self.model_config.sequence_length)
                .alias("Range"),
            )
            .drop("Path", "Index", "Date")
            .drop_nulls()
        )

        # Goal is to train on one subset of stocks and validate on another subset of stocks
        unique_stocks: List[int] = df.select("Stock ID").unique().to_series().to_list()
        rng: Generator = np.random.default_rng(self.model_config.random_seed)
        rng.shuffle(unique_stocks)

        num_stocks: int = len(unique_stocks)
        train_cut: int = int(num_stocks * self.model_config.train_fraction)
        val_cut: int = int(
            num_stocks
            * (self.model_config.train_fraction + self.model_config.val_fraction)
        )

        train_stocks: Set[int] = set(unique_stocks[:train_cut])
        val_stocks: Set[int] = set(unique_stocks[train_cut:val_cut])
        test_stocks: Set[int] = set(unique_stocks[val_cut:])

        train_df: pl.DataFrame = df.filter(pl.col("Stock ID").is_in(train_stocks))
        val_df: pl.DataFrame = df.filter(pl.col("Stock ID").is_in(val_stocks))
        test_df: pl.DataFrame = df.filter(pl.col("Stock ID").is_in(test_stocks))

        print(
            f"Num Stocks: Train={len(train_stocks)}, Val={len(val_stocks)}, Test={len(test_stocks)}"
        )
        print(
            f"Num Rows: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}"
        )

        return SplitDFDatasets(train_df, val_df, test_df)

    def _create_dataloader(self: Self, df: pl.DataFrame) -> DataLoader:
        return DataLoader(
            NASDAQDataset(df),
            batch_size=self.model_config.batch_size,
            num_workers=1,
            persistent_workers=True,
        )

    def _create_dataloaders(
        self: Self, train_df: pl.DataFrame, val_df: pl.DataFrame, test_df: pl.DataFrame
    ) -> NASDAQDataLoaders:
        print("Creating training data loader...")
        train_loader = self._create_dataloader(train_df)

        print("Creating validation data loader...")
        val_loader = self._create_dataloader(val_df)

        print("Creating test data loader...")
        test_loader = self._create_dataloader(test_df)

        print("Finished creating PyTorch DataLoaders!")

        return NASDAQDataLoaders(train_loader, val_loader, test_loader)

    def create_data_loaders_from(
        self: Self,
        data_directory: Path,
        save_directory: Path,
        dataset_loading_config: DatasetLoadingConfig = DatasetLoadingConfig.IGNORE,
    ) -> NASDAQDataLoaders:
        dataset_directory = save_directory / "scaled_datasets"
        reuse_dataset: bool = False

        if dataset_directory.exists():
            match dataset_loading_config:
                case DatasetLoadingConfig.IGNORE:
                    dataset_directory = DatasetLoadingConfig.create_unique_path_from(
                        dataset_directory
                    )
                    print(f"Training dataset will be saved to: {dataset_directory}")
                case DatasetLoadingConfig.REPLACE:
                    print(
                        f"Removing existing training dataset directory: {dataset_directory}"
                    )
                    if dataset_directory.exists():
                        shutil.rmtree(dataset_directory)
                case DatasetLoadingConfig.REUSE:
                    reuse_dataset = True
                case DatasetLoadingConfig.STOP:
                    raise FileExistsError(f"'{dataset_directory}' already exists.")

        if reuse_dataset:
            print("Loading datasets from memory...")

            train_df: pl.DataFrame = pl.read_parquet(
                dataset_directory / "train.parquet"
            )
            val_df: pl.DataFrame = pl.read_parquet(dataset_directory / "val.parquet")
            test_df: pl.DataFrame = pl.read_parquet(dataset_directory / "test.parquet")

            if train_df.is_empty() or val_df.is_empty() or test_df.is_empty():
                raise ValueError(
                    f"{RED}Cannot create datasets as some of the datasets are empty. "
                    "Try recreating the dataset by running 'self.download_dataset()' with "
                    f"dataset_loading_config=NASDAQDatasetCreationOptions.REPLACE{RESET}"
                )

            return self._create_dataloaders(train_df, val_df, test_df)

        datasets: SplitDFDatasets = self._create_datasets_from(data_directory)

        train_df = datasets.train
        val_df = datasets.val
        test_df = datasets.test

        dataset_directory.mkdir(parents=True, exist_ok=True)
        print("Saving datasets...")
        train_df.write_parquet(dataset_directory / "train.parquet")
        val_df.write_parquet(dataset_directory / "val.parquet")
        test_df.write_parquet(dataset_directory / "test.parquet")

        return self._create_dataloaders(train_df, val_df, test_df)
