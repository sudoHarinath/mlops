from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel


class DatasetConfig(BaseModel):
    name: str
    sample_size: int
    seed: int = 42


class TrainConfig(BaseModel):
    use_unsloth: bool
    max_steps: int
    batch_size: int
    learning_rate: float
    lora_r: int
    lora_alpha: int
    max_seq_length: int
    output_dir: Path


class WandbConfig(BaseModel):
    mode: Literal["online", "offline", "disabled"]
    project: str


class Config(BaseModel):
    environment: Literal["local", "remote"]
    model_name: str
    dataset: DatasetConfig
    train: TrainConfig
    wandb: WandbConfig


def load_config(profile: str) -> Config:
    path = Path("conf") / f"{profile}.yaml"
    with path.open() as f:
        data = yaml.safe_load(f)
    return Config(**data)
