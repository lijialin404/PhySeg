"""Training configurations for the Refiner and first-stage LoRA."""

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    data_root: str = str(ROOT / "PhySegD-main")
    checkpoint: str = str(ROOT / "checkpoints/physeg.pt")
    source_lora_checkpoint: str = str(ROOT / "checkpoints/physeg.pt")
    output_dir: str = str(ROOT / "runs/training")
    image_size: int = 1024
    batch_size: int = 1
    accumulation_steps: int = 1
    epochs: int = 15
    num_workers: int = 2
    cache_size: int = 2
    prefetch_factor: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    cosine_epochs: int = 15
    minimum_learning_rate: float = 1e-6
    grad_clip: float = 1.0
    mask_threshold: float = 0.0
    box_noise: float = 0.1
    box_noise_bound: int = 20
    seed: int = 20260825
    focal_weight: float = 1.0
    dice_weight: float = 1.0
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    refiner_width: int = 64
    context_scale: float = 1.25
    max_logit_residual: float = 8.0


@dataclass
class CommonConfig:
    data_root: str = str(ROOT / "PhySegD-main")
    checkpoint: str = str(ROOT / "checkpoints/physeg.pt")
    output_dir: str = ""
    image_size: int = 1024
    batch_size: int = 1
    accumulation_steps: int = 1
    num_workers: int = 2
    prefetch_factor: int = 4
    grad_clip: float = 1.0
    mask_threshold: float = 0.0
    box_noise: float = 0.1
    box_noise_bound: int = 20


@dataclass
class LoRAConfig(CommonConfig):
    model_size: str = "large"
    epochs: int = 15
    cache_size: int = 8
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.05
    seed: int = 20260909
    focal_weight: float = 20.0
    dice_weight: float = 1.0
    iou_weight: float = 1.0
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    iou_use_l1_loss: bool = True
