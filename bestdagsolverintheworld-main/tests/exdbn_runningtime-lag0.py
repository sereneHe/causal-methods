from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

# ---------------------------------------------------------------------
# Repo path
# ---------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dagsolvers.data_generation_loading_utils import load_problem_dict
from dagsolvers.dagsolver_utils import ExDagDataException
from dagsolvers.metrics_utils import count_accuracy
from dagsolvers.solve_milp import solve as solve_milp

# ---------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------
def _env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return default if v in (None, "") else v


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    return default if v in (None, "") else int(v)


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    return default if v in (None, "") else float(v)


def _env_int_list(name: str, default_csv: str) -> list[int]:
    raw = _env_str(name, default_csv)
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------
def _binary_adj_from_weights(W: np.ndarray, threshold: float = 1e-8) -> np.ndarray:
    return (np.abs(W) > threshold).astype(int)


def _ensure_dirs(out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    edges_dir = out_dir / "edges"
    edges_dir.mkdir(exist_ok=True)
    return out_dir, edges_dir


# ---------------------------------------------------------------------
# MILP config
# ---------------------------------------------------------------------
def _make_milp_cfg():
    cfg = OmegaConf.create({})
    cfg.time_limit = _env_int("EXDBN_TIME_LIMIT", 18000)
    cfg.constraints_mode = _env_str("EXDBN_CONSTRAINTS_MODE", "weights")
    cfg.callback_mode = _env_str("EXDBN_CALLBACK_MODE", "all_cycles")
    cfg.lambda1 = _env_float("EXDBN_LAMBDA1", 1.0)
    cfg.lambda2 = _env_float("EXDBN_LAMBDA2", 1.0)
    cfg.loss_type = _env_str("EXDBN_LOSS_TYPE", "l2")
    cfg.reg_type = _env_str("EXDBN_REG_TYPE", "l1")
    cfg.a_reg_type = _env_str("EXDBN_A_REG_TYPE", "l1")
    cfg.robust = _env_str("EXDBN_ROBUST", "0") == "1"
    cfg.weights_bound = _env_float("EXDBN_WEIGHTS_BOUND", 100.0)
    cfg.target_mip_gap = _env_float("EXDBN_TARGET_MIP_GAP", 0.001)
    return cfg


# ---------------------------------------------------------------------
# Data generation (ONLY n=2000)
# ---------------------------------------------------------------------
def generate_datasets(out_dir: Path, is_dynamic: bool):
    """
    Generate synthetic datasets once with n=2000.
    Static: ER graph
    Dynamic: DBN
    """

    features_list = _env_int_list(
        "EXDBN_FEATURES", "5,10,15,20,25,30,35"
    )
    n = _env_int("EXDBN_GEN_NUMBER_OF_SAMPLES", 2000)

    if is_dynamic:
        problem_name = "dynamic"
        file_prefix = "dynamic"
    else:
        #problem_name = "er"          # ✅ static 用 ER
        problem_name = "dynamic"
        Number_of_lags =0
        file_prefix = "static"

    out_dir.mkdir(parents=True, exist_ok=True)

    for d in features_list:
        print(f"[GEN] {file_prefix}: d={d}, n={n}")

        problem_cfg = {
            'name': 'dynamic',
            'generator': 'notears',
            'number_of_lags': 2,
            'number_of_variables': d,
            'number_of_samples': n,   # increase for stability
            'sem_type': 'gauss',
            'noise_scale': 1.0,
            'edge_ratio': 0.3,
            "p": 2,
            'intra_edge_ratio': 0.5,
            'inter_edge_ratio': 0.5,
            'max_data_gen_trials': 500,
            "w_max_intra": 1.0,
            "w_min_intra": 0.01,
            "w_max_inter": 0.2,
            "w_min_inter": 0.01,
            "w_decay": 1.0,
            "graph_type_intra": "er",
            "graph_type_inter": "er",
        }

        if not is_dynamic:
            problem_cfg.update(
                {
                    'number_of_lags': 0,
                }
            )

        cfg = OmegaConf.create({"problem": problem_cfg})

        problem = load_problem_dict(cfg)

        np.savez(
            out_dir / f"{file_prefix}_er_gauss_{d}.npz",
            x=problem["X"],
            y=problem["W_true"],
        )



def generate_all_datasets():
    base = Path(_env_str("EXDBN_DATA_DIR", "datasets/syntheticdata"))
    generate_datasets(base / "static", is_dynamic=False)
    generate_datasets(base / "dynamic", is_dynamic=True)


# ---------------------------------------------------------------------
# EXDBN runner (single dataset + sample size)
# ---------------------------------------------------------------------
def run_exdbn_single(npz_path: Path, out_dir: Path, sample_size: int) -> None:
    cfg = _make_milp_cfg()
    out_dir, edges_dir = _ensure_dirs(out_dir)

    raw = np.load(npz_path)
    X_full = raw["x"]
    W_true = raw["y"]
    print(W_true)
    d = X_full.shape[1]

    if X_full.shape[0] < sample_size:
        return

    X = X_full[:sample_size]
    B_true = _binary_adj_from_weights(W_true)

    runtime_file = out_dir / f"runtime_{npz_path.stem}_n{sample_size}.csv"

    # 固定列名
    columns = [
        "dataset", "features", "samples", "degree", "runtime_seconds",
        "gap", "fdr", "tpr", "fpr", "shd", "nnz", "precision", "f1score", "g_score"
    ]

    # 如果文件不存在或强制覆盖，则写入表头
    if not runtime_file.exists() or _env_str("EXDBN_OVERWRITE", "0") == "1":
        pd.DataFrame(columns=columns).to_csv(runtime_file, index=False)

    degrees = _env_int_list("EXDBN_MAX_DEGREES", "5")

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
        elapsed = time.time() - start
        if W_est is None:
            break

        B_est = _binary_adj_from_weights(W_est)
        metrics_full = count_accuracy(B_true, B_est, [], [], test_dag=True)

        # 只保留需要的指标
        metrics = {k: metrics_full.get(k, np.nan) for k in columns if k not in ["dataset", "features", "samples", "degree", "runtime_seconds", "gap"]}
        print(f"[EXDBN] {npz_path.stem} d={d} n={sample_size} deg={deg} time={elapsed:.2f}s gap={gap:.4f} metrics={metrics}")
        row = {
            "dataset": npz_path.stem,
            "features": d,
            "samples": sample_size,
            "degree": deg,
            "runtime_seconds": elapsed,
            "gap": gap,
            **metrics,
        }

        # 写入 CSV，保持列顺序固定
        pd.DataFrame([row])[columns].to_csv(runtime_file, mode="a", header=False, index=False)


# ---------------------------------------------------------------------
# Parallel dispatcher
# ---------------------------------------------------------------------
def run_exdbn_parallel():
    base_data = Path(_env_str("EXDBN_DATA_DIR", "datasets/syntheticdata"))
    base_out = Path(_env_str("EXDBN_OUT_DIR", "results/exdbn"))
    sample_sizes = _env_int_list("EXDBN_SAMPLE_SIZES", "2000")
    max_workers = _env_int("EXDBN_NUM_WORKERS", os.cpu_count() // 2)

    tasks = []

    for mode in ["static", "dynamic"]:
        for npz in (base_data / mode).glob("*.npz"):
            for n in sample_sizes:
                out_dir = base_out / f"{mode}_n{n}"
                tasks.append((npz, out_dir, n))

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(run_exdbn_single, npz, out, n)
            for npz, out, n in tasks
        ]
        for f in as_completed(futures):
            f.result()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    if _env_str("EXDBN_GENERATE_DYNAMIC", "0") == "1":
        base_dir = Path("datasets/syntheticdata")

        generate_datasets(base_dir / "static", is_dynamic=False)
        generate_datasets(base_dir / "dynamic", is_dynamic=True)
        return

    print("Running EXDBN in parallel...")
    run_exdbn_parallel()
    print("=== Done ===")


if __name__ == "__main__":
    main()
