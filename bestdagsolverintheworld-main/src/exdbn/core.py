from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd

sys.path.append('/Users/xiaoyuhe/Causal methods/bestdagsolverintheworld-main')

from dagsolvers.data_generation_loading_utils import load_problem_dict
from dagsolvers.metrics_utils import count_accuracy
from dagsolvers.solve_milp import solve as solve_milp

from .config import MilpConfig, make_milp_cfg


def _binary_adj_from_weights(W, threshold=1e-8):
    return (abs(W) > threshold).astype(int)


def generate_datasets(out_dir: Path, is_dynamic: bool, features: list[int], n: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    file_prefix = "dynamic" if is_dynamic else "static"

    for d in features:
        problem_cfg = {
            "name": "dynamic",
            "generator": "notears",
            "number_of_lags": 2 if is_dynamic else 0,
            "number_of_variables": d,
            "number_of_samples": n,
            "sem_type": "gauss",
            "noise_scale": 1.0,
            "edge_ratio": 0.3,
        }

        cfg = {"problem": problem_cfg}
        problem = load_problem_dict(cfg)

        np.savez(
            out_dir / f"{file_prefix}_er_gauss_{d}.npz",
            x=problem["X"],
            y=problem["W_true"],
        )


def run_exdbn_single(
    npz_path: Path,
    out_dir: Path,
    sample_size: int,
    degrees: list[int],
    milp_cfg: MilpConfig,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = make_milp_cfg(milp_cfg)

    raw = np.load(npz_path)
    X_full, W_true = raw["x"], raw["y"]

    if X_full.shape[0] < sample_size:
        return

    X = X_full[:sample_size]
    B_true = _binary_adj_from_weights(W_true)

    rows = []

    for deg in degrees:
        start = time.time()
        W_est, _, gap, _, _ = solve_milp(
            X=X,
            cfg=cfg,
            w_threshold=0,
            Y=[],
            B_ref=B_true,
            max_in_degree=deg,
            max_out_degree=deg,
        )
        runtime = time.time() - start

        if W_est is None:
            break

        B_est = _binary_adj_from_weights(W_est)
        metrics = count_accuracy(B_true, B_est, [], [], test_dag=True)

        rows.append(
            {
                "dataset": npz_path.stem,
                "features": X.shape[1],
                "samples": sample_size,
                "degree": deg,
                "runtime": runtime,
                "gap": gap,
                **metrics,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"{npz_path.stem}_n{sample_size}.csv", index=False)
