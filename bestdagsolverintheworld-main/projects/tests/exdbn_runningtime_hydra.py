from __future__ import annotations

import sys
import time
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

# ---------------------------------------------------------------------
# Repo root (必须在 dagsolvers import 之前)
# ---------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dagsolvers.data_generation_loading_utils import load_problem_dict
from dagsolvers.metrics_utils import count_accuracy
from dagsolvers.solve_milp import solve as solve_milp


# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------
def binary_adj(W: np.ndarray, threshold: float = 1e-8) -> np.ndarray:
    return (np.abs(W) > threshold).astype(int)


def ensure_dirs(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "edges").mkdir(exist_ok=True)


# ---------------------------------------------------------------------
# MILP config
# ---------------------------------------------------------------------
def make_milp_cfg(cfg: DictConfig):
    return OmegaConf.create(cfg.milp)


# ---------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------
def generate_datasets(cfg: DictConfig, out_dir: Path, is_dynamic: bool):
    """
    生成 synthetic datasets
    """
    for d in cfg.data.features:
        dataset_type = "dynamic" if is_dynamic else "static"
        print(f"[GEN] {dataset_type} d={d}")

        problem_cfg = {
            "name": "dynamic" if is_dynamic else "er",
            "number_of_variables": d,
            "number_of_samples": cfg.data.gen_number_of_samples,
            "noise_scale": 1.0,
        }

        if is_dynamic:
            problem_cfg.update({
                "p": 2,
                "generator": "notears",
                "graph_type_intra": "er",
                "graph_type_inter": "er",
                "intra_edge_ratio": 0.5,
                "inter_edge_ratio": 0.5,
                "w_max_intra": 1.0,
                "w_min_intra": 0.01,
                "w_max_inter": 0.2,
                "w_min_inter": 0.01,
                "w_decay": 1.0,
            })
        else:
            problem_cfg.update({
                "sem_type": "gauss",
                "edge_ratio": 2.0 / d,
            })

        problem = load_problem_dict(OmegaConf.create({"problem": problem_cfg}))

        np.savez(
            out_dir / f"{dataset_type}_er_gauss_{d}.npz",
            x=problem["X"],
            y=problem["W_true"],
        )


# ---------------------------------------------------------------------
# Single run (dataset_type, sample_size, degree)
# ---------------------------------------------------------------------
def run_single(cfg: DictConfig):
    dataset_type = cfg.data.dataset_type
    sample_size = cfg.data.sample_size
    max_degree = cfg.data.max_degree

    data_dir = Path(cfg.paths.data_dir) / dataset_type
    out_dir = Path(cfg.paths.out_dir) / f"{dataset_type}_n{sample_size}"
    ensure_dirs(out_dir)

    milp_cfg = make_milp_cfg(cfg)

    # 固定列顺序
    columns = [
        "dataset", "features", "samples", "degree",
        "runtime_seconds", "gap",
        "fdr", "tpr", "fpr", "shd", "nnz",
        "precision", "f1score", "g_score",
    ]

    for npz_path in data_dir.glob("*.npz"):
        raw = np.load(npz_path)
        X_full, W_true = raw["x"], raw["y"]

        if X_full.shape[0] < sample_size:
            continue

        X = X_full[:sample_size]
        B_true = binary_adj(W_true)
        features = X.shape[1]

        csv_path = out_dir / f"runtime_{npz_path.stem}_n{sample_size}.csv"
        if cfg.overwrite or not csv_path.exists():
            pd.DataFrame(columns=columns).to_csv(csv_path, index=False)

        start = time.time()
        W_est, _, gap, _, _ = solve_milp(
            X=X,
            cfg=milp_cfg,
            w_threshold=0,
            Y=[],
            B_ref=B_true,
            max_in_degree=max_degree,
            max_out_degree=max_degree,
        )
        elapsed = time.time() - start

        if W_est is None:
            print(f"[SKIP] {npz_path.stem} degree={max_degree}")
            continue

        metrics_full = count_accuracy(B_true, binary_adj(W_est), [], [], test_dag=True)

        # 只保留需要的列
        metrics = {k: metrics_full.get(k, np.nan) for k in columns if k not in ["dataset","features","samples","degree","runtime_seconds","gap"]}

        row = {
            "dataset": npz_path.stem,
            "features": features,
            "samples": sample_size,
            "degree": max_degree,
            "runtime_seconds": elapsed,
            "gap": gap,
            **metrics
        }

        pd.DataFrame([row])[columns].to_csv(csv_path, mode="a", header=False, index=False)

        print(f"[EXDBN] {dataset_type} | {npz_path.stem} "
              f"features={features} n={sample_size} deg={max_degree} "
              f"time={elapsed:.2f}s gap={gap:.4f}")


# ---------------------------------------------------------------------
# Hydra main
# ---------------------------------------------------------------------
@hydra.main(version_base=None, config_path="../configs", config_name="exdbn")
def main(cfg: DictConfig):
    if cfg.mode.generate_dynamic:
        base = Path(cfg.paths.data_dir)
        generate_datasets(cfg, base / "static", is_dynamic=False)
        generate_datasets(cfg, base / "dynamic", is_dynamic=True)
        return

    run_single(cfg)


if __name__ == "__main__":
    main()
