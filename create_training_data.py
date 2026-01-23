import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Self, Tuple

import polars as pl
import torch
from torch.utils.data.dataloader import DataLoader

from stocks import NASDAQDataLoaders, NASDAQDataset

# Initialize important variables
RANDOM_SEED: int = 1290
SEQUENCE_LENGTH: int = 30
TRAIN_FRACTION: float = 0.8
BATCH_SIZE: int = 256
VAL_FRACTION: float = 0.1

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


# Gemini suggested function
def check_data_integrity(df: pl.DataFrame, name: str = "Dataset") -> None:
    print(f"--- Checking {name} for NaNs and Infs ---")

    # Identify float columns (only they can hold NaN/Inf)
    float_cols: List[str] = [
        col
        for col, dtype in zip(df.columns, df.dtypes)
        if dtype in (pl.Float32, pl.Float64)
    ]

    found_issue: bool = False

    # 1. Check for Nulls
    null_counts: Tuple = df.select(pl.all().null_count()).row(0)
    if sum(null_counts) > 0:
        print(f"{RED}Found NULLs in {name}:{RESET}")
        for col, count in zip(df.columns, null_counts):
            if count > 0:
                print(f"  {col}: {count}")
        found_issue = True

    # 2. Check for NaNs
    if float_cols:
        nan_exprs: List[pl.Expr] = [
            pl.col(c).is_nan().sum().alias(c) for c in float_cols
        ]
        nan_counts: Tuple = df.select(nan_exprs).row(0)

        if sum(nan_counts) > 0:
            print(f"{RED}Found NaNs in {name}:{RESET}")
            for col, count in zip(float_cols, nan_counts):
                if count > 0:
                    print(f"  {col}: {count}")
            found_issue = True

    # 3. Check for Infs
    if float_cols:
        inf_exprs: List[pl.Expr] = [
            pl.col(c).is_infinite().sum().alias(c) for c in float_cols
        ]
        inf_counts: Tuple = df.select(inf_exprs).row(0)

        if sum(inf_counts) > 0:
            print(f"{RED}Found Infs in {name}:{RESET}")
            for col, count in zip(float_cols, inf_counts):
                if count > 0:
                    print(f"  {col}: {count}")
            found_issue = True

    if not found_issue:
        print(f"✅ {name} looks clean.")
    else:
        print(f"{RED}!! Data integrity issues found in {name} !!{RESET}")


# Gemini suggestion to improve dimensional accuracy
# RSI calculates if the stock was overbought
def compute_rsi(expr: pl.Expr, period: int = 14) -> pl.Expr:
    delta: pl.Expr = expr.diff()
    up: pl.Expr = delta.clip(lower_bound=0)
    down: pl.Expr = -delta.clip(upper_bound=0)
    # Wilder's Smoothing: alpha = 1/period. In polars ewm_mean, com = 1/alpha - 1
    # So com = period - 1
    ma_up: pl.Expr = up.ewm_mean(com=period - 1, ignore_nulls=True)
    ma_down: pl.Expr = down.ewm_mean(com=period - 1, ignore_nulls=True)
    rs: pl.Expr = ma_up / ma_down

    rsi: pl.Expr = 100 - (100 / (1 + rs))
    # Make NaNs neutral (50)
    return rsi.fill_nan(50)


def _create_datasets_from(directory: Path) -> SplitDFDatasets:
    print(f"Scanning for CSVs in {directory}...")

    window_size: int = SEQUENCE_LENGTH - 1
    glob: List = list(directory.glob("*.csv"))
    print(f"Going through {len(glob)} CSVs...")
    schema_overrides: Dict[str, type] = {
        "Open": pl.Float64,
        "High": pl.Float64,
        "Low": pl.Float64,
        "Close": pl.Float64,
        "Volume": pl.Float64,
    }

    lf: pl.LazyFrame = (
        pl.scan_csv(
            glob,
            include_file_paths="Path",
            try_parse_dates=True,
            schema_overrides=schema_overrides,
        )
        .sort(["Path", "Date"])
        # We must remove rows where Volume = 0 as log(0) = -inf
        .filter(~pl.any_horizontal(pl.col("Volume") == 0))
        .with_columns(
            pl.int_range(pl.len()).over("Path").alias("Index"),
            pl.col("Close").log().diff().over("Path").alias("Log Return"),
            (pl.col("Open").log() - pl.col("Close").log().shift(1).over("Path")).alias(
                "Log Overnight"
            ),
            (pl.col("High").log() - pl.col("Low").log()).alias("Log Range"),
            pl.col("Volume").log().diff().over("Path").alias("Log Volume Change"),
        )
        .with_columns(
            compute_rsi(pl.col("Close"), period=14).over("Path").alias("RSI"),
            (pl.col("Close").ewm_mean(span=12) - pl.col("Close").ewm_mean(span=26))
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
        .with_columns(pl.col("MACD").ewm_mean(span=9).over("Path").alias("MACD Signal"))
        .with_columns(
            pl.col("Log Return")
            .rolling_std(window_size=SEQUENCE_LENGTH)
            .over("Path")
            .alias("Rolling STD")
        )
        .with_columns(pl.all().exclude("Path", "Date", "Index").cast(pl.Float32))
        .drop("High", "Low", "Date", "Close", "Volume", "Open")
        .fill_nan(None)
        .drop_nulls()
        .filter(pl.all_horizontal(pl.all().exclude("Path", "Index").is_finite()))
        .rolling(
            index_column="Index",
            period=f"{SEQUENCE_LENGTH}i",
            group_by=pl.col("Path"),
        )
        .having(pl.len() == SEQUENCE_LENGTH)
        .all()
        .with_columns(
            pl.col("Log Return").list.last().alias("Target"),
            pl.col("Log Return").list.slice(0, window_size).alias("Close"),
            pl.col("Log Overnight").list.slice(0, window_size).alias("Open"),
            pl.col("Log Volume Change").list.slice(0, window_size).alias("Volume"),
            pl.col("Log Range").list.slice(0, window_size).alias("Range"),
            pl.col("Rolling STD").list.slice(0, window_size).alias("Rolling STD"),
            pl.col("RSI").list.slice(0, window_size),
            pl.col("MACD").list.slice(0, window_size),
            pl.col("MACD Signal").list.slice(0, window_size),
            pl.col("SMA Ratio").list.slice(0, window_size).alias("SMA Ratio"),
        )
        .drop("Log Return", "Log Overnight", "Log Range", "Log Volume Change")
        .drop_nulls()
    )

    # Sets up code for splitting up the dataset into train, val, and test datasets
    # by finding how much data points exist for each company. Then, we figure out
    # how much data points should be in the train and val datasets.
    # Test dataset gets the rest of the points. We don't sort the data before doing
    # all of these operations as the data should already be sorted from above.
    # (.rolling() assumes the data is sorted, anyway.)
    df: pl.DataFrame = lf.collect().sort(["Path", "Index"])

    all_paths: pl.DataFrame = (
        df.select("Path").unique().sample(fraction=1.0, shuffle=True, seed=RANDOM_SEED)
    )

    # This splits the dataset by ticker, so some companies may be excluded from some datasets
    # This forces to model to generalize
    n_stocks: int = len(all_paths)
    train_end = int(n_stocks * TRAIN_FRACTION)
    val_end = int(n_stocks * (TRAIN_FRACTION + VAL_FRACTION))

    train_paths: pl.DataFrame = all_paths[:train_end]
    val_paths: pl.DataFrame = all_paths[train_end:val_end]
    test_paths: pl.DataFrame = all_paths[val_end:]

    columns_to_drop: List[str] = ["Path", "Index"]
    train_df: pl.DataFrame = df.join(train_paths, on="Path", how="inner").drop(
        columns_to_drop
    )
    val_df: pl.DataFrame = df.join(val_paths, on="Path", how="inner").drop(
        columns_to_drop
    )
    test_df: pl.DataFrame = df.join(test_paths, on="Path", how="inner").drop(
        columns_to_drop
    )

    print(
        f"Created datasets: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}"
    )

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

    def standardize_column(col_name: str) -> pl.Expr:
        return (pl.col(col_name) - mean) / std

    train = train.with_columns([standardize_column(c) for c in col_group])
    val = val.with_columns([standardize_column(c) for c in col_group])
    test = test.with_columns([standardize_column(c) for c in col_group])

    normalization_data: NormalizationData = NormalizationData(mean, std)

    return train, val, test, normalization_data


def _create_dataloader(df: pl.DataFrame) -> DataLoader:
    return DataLoader(
        NASDAQDataset(df),
        batch_size=BATCH_SIZE,
        num_workers=4,
        persistent_workers=True,
    )


def _create_data_loaders(
    train_df: pl.DataFrame, val_df: pl.DataFrame, test_df: pl.DataFrame
) -> NASDAQDataLoaders:
    print("Creating training data loader...")
    train_loader = _create_dataloader(train_df)

    print("Creating validation data loader...")
    val_loader = _create_dataloader(val_df)

    print("Creating test data loader...")
    test_loader = _create_dataloader(test_df)

    print("Finished creating PyTorch DataLoaders!")

    return NASDAQDataLoaders(train_loader, val_loader, test_loader)


def create_data_loaders_from(
    directory: Path,
    save_directory: Path,
    load_datasets_from_memory: bool = False,
) -> Tuple[NASDAQDataLoaders, NormalizationData]:
    dataset_directory = save_directory / "scaled_datasets"
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
        train_df, val_df, test_df, ["Range", "Rolling STD"]
    )

    # Group 4: Oscillators (RSI is 0-100, scale it)
    train_df, val_df, test_df, _ = fit_and_scale_data(
        train_df, val_df, test_df, ["RSI"]
    )
    # Group 5: MACD
    train_df, val_df, test_df, _ = fit_and_scale_data(
        train_df, val_df, test_df, ["MACD", "MACD Signal"]
    )

    dataset_directory.mkdir(parents=True, exist_ok=True)
    print("Saving datasets...")
    train_df.write_parquet(dataset_directory / "train.parquet")
    val_df.write_parquet(dataset_directory / "val.parquet")
    test_df.write_parquet(dataset_directory / "test.parquet")
    price_norm.write_to_disc(norm_file_path)

    return _create_data_loaders(train_df, val_df, test_df), price_norm


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
    global RANDOM_SEED, SEQUENCE_LENGTH, TRAIN_FRACTION, VAL_FRACTION, BATCH_SIZE

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
