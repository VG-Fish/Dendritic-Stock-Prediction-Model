import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Set, Union

RED: str = "\033[31m"
GREEN: str = "\033[32m"
RESET: str = "\033[0m"

REQUIRED_KEYS: set[str] = {
    "sequence_length",
    "train_fraction",
    "val_fraction",
    "batch_size",
    "learning_rate",
    "epochs",
    "model_info_dir",
}


@dataclass
class ModelConfig:
    random_seed: int
    sequence_length: int
    train_fraction: float
    val_fraction: float
    batch_size: int
    learning_rate: float
    epochs: int
    load_dataset_from_memory: bool
    model_info_dir: Path


def get_config_from_json(config_path: Union[str, Path]) -> Optional[ModelConfig]:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"{RED}{config_path} doesn't exist!{RESET}")

    with open(config_path, "r") as f:
        try:
            config: Dict = json.load(f)

            missing: Set[str] = REQUIRED_KEYS - set(config.keys())
            if missing:
                raise ValueError(f"Missing key(s): {missing}")

            if config["train_fraction"] + config["val_fraction"] > 1.0:
                raise ValueError(
                    f"{RED}'train_fraction' + 'val_fraction' must be less than 1.0.{RESET}"
                )
            config["model_info_dir"] = Path(config["model_info_dir"])

            print(f"{GREEN}Parsed config file successively!{RESET}")
            return ModelConfig(
                random_seed=config.get("random_seed", 0),
                sequence_length=config["sequence_length"],
                train_fraction=config["train_fraction"],
                val_fraction=config["val_fraction"],
                batch_size=config["batch_size"],
                learning_rate=config["learning_rate"],
                epochs=config["epochs"],
                load_dataset_from_memory=config.get("load_dataset_from_memory", False),
                model_info_dir=config["model_info_dir"],
            )
        except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as e:
            print(
                f"{RED}This error occurred while trying to parse config file: {e}{RESET}"
            )
