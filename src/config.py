import torch
from dataclasses import dataclass

@dataclass
class Config:
    epochs: int
    device: torch.device
    seed: int
    sample_size: float
    learning_rate: float
    batch_size: int
    model_name: str
    wandb: bool