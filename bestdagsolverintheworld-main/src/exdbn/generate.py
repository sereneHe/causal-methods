from pathlib import Path
from .core import generate_datasets


def generate_all(
    base_dir: Path,
    features: list[int],
    samples: int,
):
    generate_datasets(base_dir / "static", False, features, samples)
    generate_datasets(base_dir / "dynamic", True, features, samples)
