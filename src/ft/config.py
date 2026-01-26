import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass(frozen=True)
class TrainConfig:
    base_model: str = "google/flan-t5-small"  # CPU-friendly demo model
    out_dir: Path = Path("runs/latest")
    max_source_len: int = 384
    max_target_len: int = 128
    train_batch_size: int = 2
    grad_accum: int = 8
    epochs: int = 3
    lr: float = 2e-4
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = ("q", "v")
    seed: int = 42


def load_manifest(run_dir: Path) -> Dict:
    path = run_dir / "manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
