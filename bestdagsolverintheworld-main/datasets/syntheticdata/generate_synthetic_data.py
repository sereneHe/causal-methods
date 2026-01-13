"""Synthetic static ER data generator.

Implements the interface you asked for:

    _ts = Generate_Synthetic_Data(File_PATH, num_datasets, T, method, sem_type, nodes, edges, noise_scale)
    _ts.genarate_data()

Notes:
- Uses `dagsolvers.data_generation_loading_utils.load_problem_dict` (notears-style ER generator).
- Saves `.npz` files with keys:
  - `x`: samples array of shape (T, d)
  - `y`: weighted adjacency matrix of shape (d, d)

"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
from omegaconf import OmegaConf

from dagsolvers.data_generation_loading_utils import load_problem_dict
from notears import utils as notears_utils


@dataclass
class Generate_Synthetic_Data:
    file_path: str
    num_datasets: int
    T: int
    method: str
    sem_type: str
    nodes: Iterable[int]
    edges: Iterable[int]
    noise_scale: float

    seed: int = 1
    overwrite: bool = False

    def __init__(
        self,
        File_PATH: str,
        num_datasets: int,
        T: int,
        method: str,
        sem_type: str,
        nodes: Iterable[int],
        edges: Iterable[int],
        noise_scale: float,
        *,
        file_path: str | None = None,
        seed: int = 1,
        overwrite: bool = False,
    ) -> None:
        # Support the exact parameter naming used in the user-provided snippet
        # while also allowing a conventional `file_path=` keyword.
        self.file_path = str(file_path) if file_path is not None else str(File_PATH)
        self.num_datasets = int(num_datasets)
        self.T = int(T)
        self.method = str(method)
        self.sem_type = str(sem_type)
        self.nodes = nodes
        self.edges = edges
        self.noise_scale = float(noise_scale)
        self.seed = int(seed)
        self.overwrite = bool(overwrite)

    def _iter_settings(self) -> List[tuple[int, int]]:
        settings: List[tuple[int, int]] = []
        for d in list(self.nodes):
            for e in list(self.edges):
                settings.append((int(d), int(e)))
        return settings

    def genarate_data(self) -> None:
        if self.method != "linear":
            raise ValueError(f"Only method='linear' is supported (got {self.method!r})")
        if self.sem_type != "gauss":
            raise ValueError(f"Only sem_type='gauss' is supported (got {self.sem_type!r})")

        out_dir = Path(self.file_path)
        out_dir.mkdir(parents=True, exist_ok=True)

        settings = self._iter_settings()
        if not settings:
            raise ValueError("nodes/edges produced no settings")

        for idx in range(int(self.num_datasets)):
            d, e = settings[idx % len(settings)]

            # `load_problem` expects edge_ratio where s0 = edge_ratio * d.
            # Here `edges` is interpreted as expected edge count (s0), so edge_ratio = s0 / d.
            edge_ratio = float(e) / float(d)

            seed = int(self.seed) + idx
            notears_utils.set_random_seed(seed)

            cfg = OmegaConf.create(
                {
                    "problem": {
                        "name": "er",
                        "number_of_variables": int(d),
                        "number_of_samples": int(self.T),
                        "edge_ratio": edge_ratio,
                        "sem_type": "gauss",
                        "noise_scale": float(self.noise_scale),
                        "normalize": False,
                    }
                }
            )

            out = load_problem_dict(cfg)
            x = np.asarray(out["X"], dtype=float)
            y = np.asarray(out["W_true"], dtype=float)

            fname = f"static_er_gauss_d{d}_e{e}_T{self.T}_seed{seed}.npz"
            fpath = out_dir / fname

            if fpath.exists() and not self.overwrite:
                # Skip rather than failing; keep generation moving.
                continue

            np.savez_compressed(fpath, x=x, y=y)


def main() -> None:
    # Example matching your snippet
    method = "linear"
    sem_type = "gauss"
    nodes = range(6, 12, 3)
    edges = range(10, 20, 5)
    T = 200
    num_datasets = 120
    File_PATH = "../Test/Examples/Test_data/"
    noise_scale = 1.0

    _ts = Generate_Synthetic_Data(File_PATH, num_datasets, T, method, sem_type, nodes, edges, noise_scale)
    _ts.genarate_data()


if __name__ == "__main__":
    main()
