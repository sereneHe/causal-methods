from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

from .core import run_exdbn_single
from .config import MilpConfig, make_milp_cfg
from dagsolvers.solve_milp import solve as solve_milp
from dagsolvers.metrics_utils import count_accuracy
import numpy as np


def run_parallel(
    mode: str,
    data_dir: Path,
    out_dir: Path,
    sample_sizes: list[int],
    degrees: list[int],
    milp_cfg: MilpConfig,
    num_workers: int,
):
    tasks = []

    for npz in (data_dir / mode).glob("*.npz"):
        for n in sample_sizes:
            tasks.append((npz, out_dir / f"{mode}_n{n}", n))

    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        for npz, out, n in tasks:
            pool.submit(
                run_exdbn_single,
                npz,
                out,
                n,
                degrees,
                milp_cfg,
            )


def run_exdbn(problem: dict, max_degree: int, milp_cfg: MilpConfig):
    X = problem.get("X", problem.get("x"))  # Handle both 'X' and 'x' keys
    B_true = problem["y"]

    cfg = make_milp_cfg(milp_cfg)  # Create the configuration object

    W_est, _, gap, _, _ = solve_milp(
        X=X,
        cfg=cfg,
        w_threshold=0,  # Set weight threshold to 0
        Y=[],
        B_ref=B_true,
        max_in_degree=max_degree,
        max_out_degree=max_degree,
    )

    B_est = (np.abs(W_est) > 1e-8).astype(int)
    metrics = count_accuracy(B_true, B_est, [], [], test_dag=True)

    return metrics, gap

