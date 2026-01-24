from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Self

import polars as pl
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


class TrainingDataCreator:
    def __init__(self: Self, model_config: ModelConfig) -> None:
        self.model_config: ModelConfig = model_config

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
        print(f"Scanning for CSVs in {directory}...")

        window_size: int = self.model_config.sequence_length - 1
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

        df: pl.DataFrame = (
            (
                pl.scan_csv(
                    glob,
                    include_file_paths="Path",
                    try_parse_dates=True,
                    schema_overrides=schema_overrides,
                )
                .sort(["Path", "Date"])
                # We make the rows where Volume = 0 equal to 1 as log(1) = 0
                .with_columns(
                    pl.col("Date").dt.date(),
                    pl.int_range(pl.len()).over("Path").alias("Index"),
                    # log1p() is better for Volume as log1p(0) = 0 & is more stable for smaller x
                    pl.col("Close").log1p().diff().over("Path").alias("Log Return"),
                    (
                        pl.col("Open").log()
                        - pl.col("Close").log().shift(1).over("Path")
                    ).alias("Log Overnight"),
                    (pl.col("High").log() - pl.col("Low").log()).alias("Log Range"),
                    pl.col("Volume")
                    .log()
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
                    pl.col("MACD").ewm_mean(span=9).over("Path").alias("MACD Signal")
                )
                .with_columns(
                    pl.col("Log Return")
                    .rolling_std(window_size=self.model_config.sequence_length)
                    .over("Path")
                    .alias("Rolling STD")
                )
                .with_columns(
                    pl.all().exclude("Path", "Date", "Index").cast(pl.Float32)
                )
                .drop("High", "Low", "Close", "Volume", "Open")
                # Remove NaNs, nulls, and infinities
                .fill_nan(None)
                .drop_nulls()
                .filter(
                    pl.all_horizontal(
                        pl.all().exclude("Path", "Index", "Date").is_finite()
                    ),
                )
            )
            .collect()
            .sort(["Path", "Index"])
            .rolling(
                index_column="Index",
                period=f"{self.model_config.sequence_length}i",
                group_by="Path",
            )
            .having(pl.len() == self.model_config.sequence_length)
            .agg(
                # This makes the data appropriate for a classification model
                (pl.col("Log Return").last() > 0).cast(pl.Float32).alias("Target"),
                pl.col("Date").last(),
                pl.col("RSI").slice(0, window_size),
                pl.col("MACD").slice(0, window_size),
                pl.col("MACD Signal").slice(0, window_size),
                pl.col("SMA Ratio").slice(0, window_size),
                pl.col("Rolling STD").slice(0, window_size),
                pl.col("Log Return").slice(0, window_size).alias("Close"),
                pl.col("Log Overnight").slice(0, window_size).alias("Open"),
                pl.col("Log Volume Change").slice(0, window_size).alias("Volume"),
                pl.col("Log Range").slice(0, window_size).alias("Range"),
            )
            .drop("Path", "Index")
        )

        date_cutoffs: pl.DataFrame = df.select(
            pl.col("Date")
            .quantile(self.model_config.train_fraction, interpolation="nearest")
            .alias("Train Cutoff"),
            pl.col("Date")
            .quantile(
                self.model_config.train_fraction + self.model_config.val_fraction,
                interpolation="nearest",
            )
            .alias("Val Cutoff"),
        )
        train_date_cutoff: date = date_cutoffs["Train Cutoff"].dt.date().item()
        val_date_cutoff: date = date_cutoffs["Val Cutoff"].dt.date().item()

        # We minus by timedelta to remove the shared data between train and val datasets that came as a result of using .rolling()
        train_df: pl.DataFrame = df.filter(
            pl.col("Date")
            < train_date_cutoff - timedelta(days=self.model_config.sequence_length)
        ).drop("Date")

        val_df: pl.DataFrame = df.filter(
            (pl.col("Date") >= train_date_cutoff)
            & (
                pl.col("Date")
                < val_date_cutoff - timedelta(days=self.model_config.sequence_length)
            )
        ).drop("Date")

        test_df: pl.DataFrame = df.filter(pl.col("Date") >= val_date_cutoff).drop(
            "Date"
        )

        print(
            f"Created datasets: Train Length={len(train_df)}, Val Length={len(val_df)}, Test Length={len(test_df)}"
        )

        return SplitDFDatasets(train_df, val_df, test_df)

    def _create_dataloader(self: Self, df: pl.DataFrame) -> DataLoader:
        return DataLoader(
            NASDAQDataset(df),
            batch_size=self.model_config.batch_size,
            num_workers=0,
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
        directory: Path,
        save_directory: Path,
        load_datasets_from_memory: bool = False,
    ) -> NASDAQDataLoaders:
        dataset_directory = save_directory / "scaled_datasets"

        if load_datasets_from_memory:
            print("Loading datasets from memory...")

            train_df: pl.DataFrame = pl.read_parquet(
                dataset_directory / "train.parquet"
            )
            val_df: pl.DataFrame = pl.read_parquet(dataset_directory / "val.parquet")
            test_df: pl.DataFrame = pl.read_parquet(dataset_directory / "test.parquet")

            return self._create_dataloaders(train_df, val_df, test_df)

        datasets: SplitDFDatasets = self._create_datasets_from(directory)

        train_df = datasets.train
        val_df = datasets.val
        test_df = datasets.test

        dataset_directory.mkdir(parents=True, exist_ok=True)
        print("Saving datasets...")
        train_df.write_parquet(dataset_directory / "train.parquet")
        val_df.write_parquet(dataset_directory / "val.parquet")
        test_df.write_parquet(dataset_directory / "test.parquet")

        return self._create_dataloaders(train_df, val_df, test_df)
