# run.py
from pathlib import Path
import time
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

from dagsolvers.solve_milp import solve as solve_milp
from dagsolvers.metrics_utils import count_accuracy

def _binary_adj_from_weights(W: np.ndarray, threshold: float = 1e-8) -> np.ndarray:
    return (np.abs(W) > threshold).astype(int)

def _ensure_dirs(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    edges_dir = out_dir / "edges"
    edges_dir.mkdir(exist_ok=True)
    return out_dir, edges_dir

def _make_milp_cfg(solver_cfg: dict):
    from omegaconf import OmegaConf
    cfg = OmegaConf.create(solver_cfg)
    return cfg

def run_exdbn_single(npz_path, out_dir, sample_size, lambda1, lambda2, max_degrees, solver_cfg):
    cfg = _make_milp_cfg(solver_cfg)
    if lambda1 is not None: cfg.lambda1 = lambda1
    if lambda2 is not None: cfg.lambda2 = lambda2

    out_dir, _ = _ensure_dirs(out_dir)
    raw = np.load(npz_path)
    X_full, W_true = raw["x"], raw["y"]

    if X_full.shape[0] < sample_size: return
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
            X=X, cfg=cfg, w_threshold=0, Y=[], B_ref=B_true,
            max_in_degree=deg, max_out_degree=deg
        )
        elapsed = time.time() - start
        if W_est is None: break

        B_est = _binary_adj_from_weights(W_est)
        metrics_full = count_accuracy(B_true, B_est, [], [], test_dag=True)
        metrics = {k: metrics_full.get(k, float('nan')) for k in columns if k not in ["dataset","features","samples","degree","runtime_seconds","gap"]}

        row = {"dataset": npz_path.stem,"features":X.shape[1],"samples":sample_size,"degree":deg,"runtime_seconds":elapsed,"gap":gap,**metrics}
        pd.DataFrame([row])[columns].to_csv(runtime_file, mode="a", header=False, index=False)
        print(f"[EXDBN] {npz_path.stem} deg={deg} time={elapsed:.2f}s metrics={metrics}")

def run_exdbn_parallel_from_cfg(cfg: dict, mode: str,
                                lambda1=None, lambda2=None,
                                sample_sizes=None, max_degrees=None,
                                num_workers=None):
    data_dir = Path(cfg["data_dir"]) / mode
    out_dir = Path(cfg["out_dir"]) / mode
    solver_cfg = cfg["solver"]
    sample_sizes = sample_sizes if sample_sizes is not None else solver_cfg["sample_sizes"]
    max_degrees = max_degrees if max_degrees is not None else solver_cfg["max_degrees"]
    num_workers = num_workers if num_workers is not None else cfg.get("num_workers", 4)

    npz_list = list(data_dir.glob("*.npz"))
    from functools import partial
    partial_run = partial(run_exdbn_single, lambda1=lambda1, lambda2=lambda2,
                          max_degrees=max_degrees, solver_cfg=solver_cfg)

    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = [pool.submit(partial_run, npz, out_dir, n) for npz in npz_list for n in sample_sizes]
        for f in futures:
            f.result()
