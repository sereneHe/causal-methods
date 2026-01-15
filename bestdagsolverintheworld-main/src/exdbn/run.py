# run.py
from pathlib import Path
import time
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from dagsolvers.solve_milp import solve as solve_milp
from dagsolvers.metrics_utils import count_accuracy

from exdbn.generate import generate_datasets

def _binary_adj_from_weights(W: np.ndarray, threshold: float = 1e-8) -> np.ndarray:
    return (np.abs(W) > threshold).astype(int)

def _ensure_dirs(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    edges_dir = out_dir / "edges"
    edges_dir.mkdir(exist_ok=True)
    return out_dir, edges_dir

def _make_milp_cfg(lambda1=1.0, lambda2=1.0):
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({})
    cfg.lambda1 = lambda1
    cfg.lambda2 = lambda2
    cfg.time_limit = 18000
    cfg.constraints_mode = "weights"
    cfg.callback_mode = "all_cycles"
    cfg.loss_type = "l2"
    cfg.reg_type = "l1"
    cfg.a_reg_type = "l1"
    cfg.robust = False
    cfg.weights_bound = 100.0
    cfg.target_mip_gap = 0.001
    return cfg

def run_exdbn_single(npz_path, out_dir, sample_size, lambda1, lambda2, max_degrees):
    cfg = _make_milp_cfg(lambda1, lambda2)
    out_dir, _ = _ensure_dirs(out_dir)

    raw = np.load(npz_path)
    X_full = raw["x"]
    W_true = raw["y"]

    if X_full.shape[0] < sample_size:
        return

    X = X_full[:sample_size]
    B_true = _binary_adj_from_weights(W_true)

    runtime_file = out_dir / f"runtime_{npz_path.stem}_n{sample_size}.csv"

    columns = [
        "dataset", "features", "samples", "degree", "runtime_seconds",
        "gap", "fdr", "tpr", "fpr", "shd", "nnz", "precision", "f1score", "g_score"
    ]

    if not runtime_file.exists():
        pd.DataFrame(columns=columns).to_csv(runtime_file, index=False)

    for deg in max_degrees:
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
        elapsed = time.time() - start
        if W_est is None:
            break

        B_est = _binary_adj_from_weights(W_est)
        metrics_full = count_accuracy(B_true, B_est, [], [], test_dag=True)
        metrics = {k: metrics_full.get(k, np.nan) for k in columns if k not in ["dataset", "features", "samples", "degree", "runtime_seconds", "gap"]}

        row = {
            "dataset": npz_path.stem,
            "features": X.shape[1],
            "samples": sample_size,
            "degree": deg,
            "runtime_seconds": elapsed,
            "gap": gap,
            **metrics,
        }
        pd.DataFrame([row])[columns].to_csv(runtime_file, mode="a", header=False, index=False)
        print(f"[EXDBN] {npz_path.stem} deg={deg} time={elapsed:.2f}s metrics={metrics}")

def run_exdbn_parallel(
    base_data: Path,
    base_out: Path,
    sample_sizes: list[int],
    max_degrees: list[int],
    lambda1: float,
    lambda2: float,
    mode: str,
    num_workers: int = None
):
    npz_list = list((base_data).glob("*.npz"))
    if num_workers is None:
        import os
        num_workers = max(1, os.cpu_count() // 2)

    tasks = []
    for npz in npz_list:
        for n in sample_sizes:
            out_dir = base_out / f"{mode}_n{n}"
            tasks.append((npz, out_dir, n, lambda1, lambda2, max_degrees))

    from functools import partial
    run_func = partial(run_exdbn_single)
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = [pool.submit(run_exdbn_single, *t) for t in tasks]
        for f in futures:
            f.result()
