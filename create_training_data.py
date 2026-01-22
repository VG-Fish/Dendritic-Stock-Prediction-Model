import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Self, Tuple

import polars as pl
import torch
from torch.utils.data.dataloader import DataLoader

from download import NASDAQDatasetInfo, NASDAQDownloader
from stocks import NASDAQDataLoaders, NASDAQDataset

# Initialize important variables
RANDOM_SEED: int = 1290
SEQUENCE_LENGTH: int = 30
TRAIN_FRACTION: float = 0.8
BATCH_SIZE: int = 256
VAL_FRACTION: float = 0.1
MODEL_INFO_DIR: Path = Path("improved_lstm_model_info")

# ANSI escape codes
RED: str = "\033[31m"
RESET: str = "\033[0m"


@dataclass
class SplitDFDatasets:
    train: pl.DataFrame
    val: pl.DataFrame
    test: pl.DataFrame


@dataclass
class NormalizationData:
    mean: float
    std: float

    def write_to_disc(self: Self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(f"mean={self.mean}\nstd={self.std}")

    @classmethod
    def read_from_disc(cls: type, path: Path) -> "NormalizationData":
        if not path.exists():
            raise FileNotFoundError(f"Normalization file not found at {path}")

        data: Dict[str, float] = {}
        with open(path, "r") as f:
            for line in f:
                if "=" in line:
                    key, val = line.strip().split("=")
                    data[key] = float(val)

        return cls(mean=data["mean"], std=data["std"])


def _create_datasets_from(directory: Path) -> SplitDFDatasets:
    print(f"Scanning for CSVs in {directory}...")

    window_size: int = SEQUENCE_LENGTH - 1
    lf: pl.LazyFrame = (
        pl.scan_csv(
            directory / "*.csv", include_file_paths="Path", try_parse_dates=True
        )
        .sort(["Path", "Date"])
        .with_columns(
            pl.int_range(pl.len()).over("Path").alias("Index"),
            (pl.col("High") - pl.col("Low")).alias("Range"),
            pl.col("Close")
            .rolling_std(window_size=SEQUENCE_LENGTH)
            .alias("Rolling STD"),
        )
        .with_columns(pl.all().exclude("Path", "Date", "Index").cast(pl.Float32))
        .drop("High", "Low", "Date")
        .drop_nulls()
        .rolling(
            index_column="Index",
            period=f"{SEQUENCE_LENGTH}i",
            group_by=pl.col("Path"),
        )
        .having(pl.len() == SEQUENCE_LENGTH)
        .all()
        .with_columns(
            pl.col("Close").list.last().alias("Target"),
            # I'm slicing the lists manually without using a loop as I couldn't get it to work
            # + this approach should be faster.
            pl.col("Close").list.slice(0, window_size),
            pl.col("Open").list.slice(0, window_size),
            pl.col("Volume").list.slice(0, window_size),
            pl.col("Range").list.slice(0, window_size),
            pl.col("Rolling STD").list.slice(0, window_size),
        )
    )
    # Sets up code for splitting up the dataset into train, val, and test datasets
    # by finding how much data points exist for each company. Then, we figure out
    # how much data points should be in the train and val datasets.
    # Test dataset gets the rest of the points. We don't sort the data before doing
    # all of these operations as the data should already be sorted from above.
    # (.rolling() assumes the data is sorted, anyway.)
    df: pl.DataFrame = lf.collect()
    df = df.with_columns(pl.len().over("Path").alias("Num Path"))
    df = df.with_columns(
        (pl.col("Num Path") * TRAIN_FRACTION).cast(pl.Int64).alias("Num Train"),
        (pl.col("Num Path") * VAL_FRACTION).cast(pl.Int64).alias("Num Val"),
    ).drop_nulls()

    columns_to_drop: List[str] = ["Path", "Index", "Num Path", "Num Train", "Num Val"]

    train_df: pl.DataFrame = df.filter(pl.col("Index") < pl.col("Num Train")).drop(
        columns_to_drop
    )

    val_df: pl.DataFrame = df.filter(
        (pl.col("Index") >= pl.col("Num Train"))
        & (pl.col("Index") < pl.col("Num Train") + pl.col("Num Val"))
    ).drop(columns_to_drop)

    test_df: pl.DataFrame = df.filter(
        pl.col("Index") >= pl.col("Num Train") + pl.col("Num Val")
    ).drop(columns_to_drop)

    print("Created initial datasets...")

    return SplitDFDatasets(train_df, val_df, test_df)


def fit_and_scale_data(
    train: pl.DataFrame, val: pl.DataFrame, test: pl.DataFrame, col_group: List[str]
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, NormalizationData]:
    print(f"Scaling columns {', '.join(col_group)} in datasets...")

    series_to_concat: List[pl.Series] = []
    for col_name in col_group:
        dtype: pl.DataType = train[col_name].dtype

        if isinstance(dtype, pl.List):
            series_to_concat.append(train[col_name].list.explode())
        else:
            series_to_concat.append(train[col_name])

    combined_data: pl.Series = pl.concat(series_to_concat)

    mean: float = float(combined_data.mean())  # pyright: ignore[reportArgumentType]
    std: float = float(combined_data.std())  # pyright: ignore[reportArgumentType]

    # Avoid division by zero
    if std == 0:
        std = 1.0

    def standardize_column(col_name: str):
        return (pl.col(col_name) - mean) / std

    train = train.with_columns([standardize_column(c) for c in col_group])
    val = val.with_columns([standardize_column(c) for c in col_group])
    test = test.with_columns([standardize_column(c) for c in col_group])

    normalization_data: NormalizationData = NormalizationData(mean, std)

    return train, val, test, normalization_data


def _create_data_loaders(
    train_df: pl.DataFrame, val_df: pl.DataFrame, test_df: pl.DataFrame
) -> NASDAQDataLoaders:
    print("Creating training data loader...")
    train_loader = DataLoader(
        NASDAQDataset(train_df), batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )

    print("Creating validation data loader...")
    val_loader = DataLoader(
        NASDAQDataset(val_df), batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    print("Creating test data loader...")
    test_loader = DataLoader(
        NASDAQDataset(test_df), batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    print("Finished creating PyTorch DataLoaders!")

    return NASDAQDataLoaders(train_loader, val_loader, test_loader)


def create_data_loaders_from(
    directory: Path, load_datasets_from_memory: bool = False
) -> Tuple[NASDAQDataLoaders, NormalizationData]:
    dataset_directory = MODEL_INFO_DIR / "scaled_datasets"
    norm_file_path = dataset_directory / "normalization_data.txt"

    if load_datasets_from_memory:
        print("Loading datasets from memory...")

        if not norm_file_path.exists():
            raise ValueError(
                f"{RED}No metadata found! Run with load_datasets_from_memory=False first.{RESET}"
            )

        train_df: pl.DataFrame = pl.read_parquet(dataset_directory / "train.parquet")
        val_df: pl.DataFrame = pl.read_parquet(dataset_directory / "val.parquet")
        test_df: pl.DataFrame = pl.read_parquet(dataset_directory / "test.parquet")
        normalization_data = NormalizationData.read_from_disc(norm_file_path)

        return _create_data_loaders(train_df, val_df, test_df), normalization_data

    datasets: SplitDFDatasets = _create_datasets_from(directory)

    train_df = datasets.train
    val_df = datasets.val
    test_df = datasets.test

    # Group 1: Open, Close, Target
    # We choose related groups to scale all of them together
    train_df, val_df, test_df, price_norm = fit_and_scale_data(
        train_df, val_df, test_df, ["Close", "Open", "Target"]
    )

    # Group 2: Volume
    train_df, val_df, test_df, _ = fit_and_scale_data(
        train_df, val_df, test_df, ["Volume"]
    )

    # Group 3: Range
    train_df, val_df, test_df, _ = fit_and_scale_data(
        train_df, val_df, test_df, ["Range"]
    )

    # Group 4: Rolling STD
    train_df, val_df, test_df, _ = fit_and_scale_data(
        train_df, val_df, test_df, ["Rolling STD"]
    )

    dataset_directory.mkdir(parents=True, exist_ok=True)
    train_df.write_parquet(dataset_directory / "train.parquet")
    val_df.write_parquet(dataset_directory / "val.parquet")
    test_df.write_parquet(dataset_directory / "test.parquet")
    price_norm.write_to_disc(norm_file_path)

    return _create_data_loaders(train_df, val_df, test_df), price_norm


def main() -> None:
    model_save_dir: Path = MODEL_INFO_DIR / "models"
    os.makedirs(model_save_dir, exist_ok=True)

    downloader: NASDAQDownloader = NASDAQDownloader()
    info: NASDAQDatasetInfo = downloader.download_dataset(stop_if_dest_dir_exists=True)

    create_data_loaders_from(info.stocks_directory, load_datasets_from_memory=False)


def float_in_range(low: float, high: float) -> Callable:
    def checker(value: str) -> float:
        try:
            f_value = float(value)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"{RED}'{value}' is not a valid float.{RESET}"
            )

        f_value = float(value)
        if f_value < low or f_value >= high:
            raise argparse.ArgumentTypeError(
                f"{RED}Value must be in [{low}, {high}){RESET}"
            )
        return f_value

    return checker


def parse_args() -> argparse.Namespace:
    global \
        RANDOM_SEED, \
        SEQUENCE_LENGTH, \
        TRAIN_FRACTION, \
        VAL_FRACTION, \
        BATCH_SIZE, \
        MODEL_INFO_DIR

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog="Dendritic LSTM Stock Prediction Model",
        description="This program trains a Dendritic LSTM Stock Prediction Model on data from thousands of companies from the NASDAQ.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--random_seed", help="Set the random seed.", type=int, default=RANDOM_SEED
    )
    parser.add_argument(
        "--sequence_length",
        help="How many days the model should consider to predict the price for the next day.",
        type=int,
        default=SEQUENCE_LENGTH,
    )
    parser.add_argument(
        "--train_fraction",
        help="How big the training dataset should be. This parameter should be in [0, 1).",
        type=float_in_range(0.0, 1.0),
        default=TRAIN_FRACTION,
    )
    parser.add_argument(
        "--val_fraction",
        help="How big the validation dataset should be. This parameter should be in [0, 1).",
        type=float_in_range(0.0, 1.0),
        default=VAL_FRACTION,
    )
    parser.add_argument(
        "--batch_size",
        help="How big each batch size should be. This parameter should ideally be a power of 2.",
        type=int,
        default=BATCH_SIZE,
    )
    parser.add_argument(
        "--model_info_dir",
        help="The directory to save all model related stuff to.",
        type=str,
        default=MODEL_INFO_DIR,
    )
    args: argparse.Namespace = parser.parse_args()

    if args.train_fraction + args.val_fraction >= 1.0:
        parser.error(
            f"{RED}The sum of --train_fraction ({args.train_fraction}) and "
            f"--val_fraction ({args.val_fraction}) must be less than 1.0 "
            f"to leave room for the test set.{RESET}"
        )
    return args


if __name__ == "__main__":
    args: argparse.Namespace = parse_args()

    RANDOM_SEED = args.random_seed
    torch.manual_seed(RANDOM_SEED)
    SEQUENCE_LENGTH = args.sequence_length
    TRAIN_FRACTION = args.train_fraction
    VAL_FRACTION = args.val_fraction
    BATCH_SIZE = args.batch_size
    MODEL_INFO_DIR = Path(args.model_info_dir)

    main()
