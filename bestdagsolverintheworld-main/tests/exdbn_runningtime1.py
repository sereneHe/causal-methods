from __future__ import annotations

import sys
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dagsolvers.data_generation_loading_utils import load_problem_dict
from dagsolvers.dagsolver_utils import ExDagDataException
from dagsolvers.metrics_utils import count_accuracy
from dagsolvers.solve_milp import solve as solve_milp


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None or value == "" else value


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None or value == "" else int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None or value == "" else float(value)


def _env_int_list(name: str, default_csv: str) -> list[int]:
    raw = _env_str(name, default_csv)
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _binary_adj_from_weights(W: np.ndarray, threshold: float = 1e-8) -> np.ndarray:
    return (np.abs(W) > threshold).astype(int)


def _ensure_results_dirs(out_dir: Path) -> tuple[Path, Path]:
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    edges_dir = out_dir / "edges"
    edges_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, edges_dir


def _make_milp_cfg():
    cfg = OmegaConf.create({})
    cfg.time_limit = _env_int("EXDBN_TIME_LIMIT", 36000)  # Configurable via EXDBN_TIME_LIMIT
    cfg.constraints_mode = _env_str("EXDBN_CONSTRAINTS_MODE", "weights")
    cfg.callback_mode = _env_str("EXDBN_CALLBACK_MODE", "all_cycles")
    cfg.lambda1 = _env_float("EXDBN_LAMBDA1", 1.0)
    cfg.lambda2 = _env_float("EXDBN_LAMBDA2", 1.0)
    cfg.loss_type = _env_str("EXDBN_LOSS_TYPE", "l2")
    cfg.reg_type = _env_str("EXDBN_REG_TYPE", "l1")
    cfg.a_reg_type = _env_str("EXDBN_A_REG_TYPE", "l1")
    cfg.robust = _env_str("EXDBN_ROBUST", "0") == "1"
    cfg.weights_bound = _env_float("EXDBN_WEIGHTS_BOUND", 100.0)
    cfg.target_mip_gap = _env_float("EXDBN_TARGET_MIP_GAP", 0.001)  # Configurable via EXDBN_TARGET_MIP_GAP
    cfg.tabu_edges = _env_str("EXDBN_TABU_EDGES", "0") == "1"
    cfg.plot_dpi = _env_int("EXDBN_PLOT_DPI", 100)
    return cfg


def generate_dynamic_datasets(out_dir: Path) -> None:
    features_list = _env_int_list("EXDBN_GEN_FEATURES", "5,10,15,20,25")  # Configurable via EXDBN_GEN_FEATURES
    is_dynamic = _env_str("EXDBN_GEN_DYNAMIC", "1") == "1"
    name = _env_str("EXDBN_GEN_NAME", "dynamic")
    generator = _env_str("EXDBN_GEN_GENERATOR", "notears")
    variant = _env_str("EXDBN_GEN_VARIANT", "er")
    p = _env_int("EXDBN_GEN_NUMBER_OF_LAGS", 2)
    if not is_dynamic:
        p = 0
    n = _env_int("EXDBN_GEN_NUMBER_OF_SAMPLES", 200)
    sem_type = _env_str("EXDBN_GEN_SEM_TYPE", "gauss")
    noise_scale = _env_float("EXDBN_GEN_NOISE_SCALE", 1.0)

    raw_edge_ratio = os.getenv("EXDBN_GEN_EDGE_RATIO")
    edge_ratio = None if raw_edge_ratio is None or raw_edge_ratio == "" else float(raw_edge_ratio)

    intra_edge_ratio = _env_float("EXDBN_GEN_INTRA_EDGE_RATIO", 0.5)
    inter_edge_ratio = _env_float("EXDBN_GEN_INTER_EDGE_RATIO", 0.5)
    if edge_ratio is not None:
        if os.getenv("EXDBN_GEN_INTRA_EDGE_RATIO") in (None, ""):
            intra_edge_ratio = float(edge_ratio)
        if os.getenv("EXDBN_GEN_INTER_EDGE_RATIO") in (None, ""):
            inter_edge_ratio = float(edge_ratio)
    w_max_intra = _env_float("EXDBN_GEN_W_MAX_INTRA", 1.0)
    w_min_intra = _env_float("EXDBN_GEN_W_MIN_INTRA", 0.01)
    w_max_inter = _env_float("EXDBN_GEN_W_MAX_INTER", 0.2)
    w_min_inter = _env_float("EXDBN_GEN_W_MIN_INTER", 0.01)
    w_decay = _env_float("EXDBN_GEN_W_DECAY", 1.0)
    max_data_gen_trials = _env_int("EXDBN_GEN_MAX_DATA_GEN_TRIALS", 1000)
    max_attempts = _env_int("EXDBN_GEN_MAX_ATTEMPTS", 5)

    out_dir.mkdir(parents=True, exist_ok=True)

    for num_nodes in features_list:
        problem_cfg: dict[str, object] = {
            "name": name,
            "generator": generator,
            "p": p,
            "number_of_variables": int(num_nodes),
            "number_of_samples": n,
            "noise_scale": noise_scale,
            "graph_type_intra": variant,
            "graph_type_inter": "er",
            "intra_edge_ratio": intra_edge_ratio,
            "inter_edge_ratio": inter_edge_ratio,
            "w_max_intra": w_max_intra,
            "w_min_intra": w_min_intra,
            "w_max_inter": w_max_inter,
            "w_min_inter": w_min_inter,
            "w_decay": w_decay,
            "sem_type": sem_type,
            "max_data_gen_trials": max_data_gen_trials,
        }
        if edge_ratio is not None:
            problem_cfg["edge_ratio"] = float(edge_ratio)

        cfg = OmegaConf.create({"problem": problem_cfg})

        last_exc: Exception | None = None
        for attempt in range(max_attempts):
            try:
                problem = load_problem_dict(cfg)
                X = problem["X"]
                W_true = problem["W_true"]
                file_name = out_dir / f"dynamic_{variant}_{sem_type}_{num_nodes}.npz"
                np.savez(file_name, x=X, y=W_true)
                print(f"Saved {file_name} with shape X={X.shape}, W_true={W_true.shape}")
                last_exc = None
                break
            except ExDagDataException as e:
                last_exc = e
                print(f"Attempt {attempt + 1}/{max_attempts} failed for d={num_nodes}: {e}")
        if last_exc is not None:
            raise last_exc


def run_exdbn_on_npz_dir(data_dir: Path, out_dir: Path) -> None:
    # Ensure output directory exists
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
    cfg = _make_milp_cfg()
    out_dir, edges_dir = _ensure_results_dirs(out_dir)

    deg_file = out_dir / "max_degree.csv"
    if not deg_file.exists():
        pd.DataFrame(columns=["dataset", "features_num", "max_in_degree", "max_out_degree"]).to_csv(
            deg_file, index=False
        )

    degrees = _env_int_list("EXDBN_MAX_DEGREES", os.getenv("EXDBN_MAX_DEGREES_DEFAULT", "5,10,15,20"))  # Configurable via EXDBN_MAX_DEGREES
    limit = _env_int("EXDBN_LIMIT_DATASETS", 0)
    overwrite = _env_str("EXDBN_OVERWRITE", "0") == "1"

    npz_files = sorted(data_dir.glob("*.npz"))
    if limit > 0:
        npz_files = npz_files[:limit]

    raw_features = os.getenv("EXDBN_FEATURES", "5")  # Default to feature 5 for testing
    allowed_features = {int(feature) for feature in raw_features.split(',')}

    # Add debug information to log missing .npz files
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}. Ensure the directory exists and contains valid .npz files.")
    # Log the names of all .npz files found
    print(f"Found .npz files: {[file.name for file in npz_files]}")

    for idx, npz_path in enumerate(npz_files, start=1):
        raw = np.load(npz_path, allow_pickle=True)
        X = raw["x"]
        W_true = raw["y"]

        d = int(X.shape[1])
        if allowed_features is not None and d not in allowed_features:
            print(f"[{idx}/{len(npz_files)}] Skipping {npz_path.name} (d={d}) not in EXDBN_FEATURES")
            continue
        B_true = _binary_adj_from_weights(W_true)

        runtime_file = out_dir / f"runtime_{npz_path.stem}.csv"
        if runtime_file.exists() and not overwrite:
            print(f"[{idx}/{len(npz_files)}] Skipping existing {runtime_file.name}")
            continue
        pd.DataFrame(
            columns=[
                "dataset",
                "features",
                "degree",
                "runtime_seconds",
                "gap",
                "lazy_count",
                "fdr",
                "tpr",
                "fpr",
                "shd",
                "nnz",
                "precision",
                "f1score",
                "g_score",
            ]
        ).to_csv(runtime_file, index=False)

        max_in_degree_test = 0
        max_out_degree_test = 0

        print(f"\n[{idx}/{len(npz_files)}] Dataset={npz_path.name} (d={d})")
        for deg in degrees:
            try:
                print(f"  Trying max_degree={deg}...")
                start_time = time.time()
                W_est, A_est, gap, lazy_count, _stats = solve_milp(
                    X=X,
                    cfg=cfg,
                    w_threshold=0,
                    Y=[],
                    B_ref=B_true,
                    max_in_degree=int(deg),
                    max_out_degree=int(deg),
                )
                elapsed = time.time() - start_time

                if W_est is None:
                    print("  Infeasible (no solution).")
                    break

                B_est = _binary_adj_from_weights(W_est)
                metrics = count_accuracy(B_true, B_est, [], [], test_dag=True)

                pd.DataFrame(B_est).to_csv(
                    edges_dir / f"{npz_path.stem}_deg{deg}.csv",
                    index=False,
                    header=False,
                )

                row = {
                    "dataset": npz_path.stem,
                    "features": d,
                    "degree": int(deg),
                    "runtime_seconds": elapsed,
                    "gap": gap,
                    "lazy_count": lazy_count,
                    **{k: metrics.get(k) for k in ["fdr", "tpr", "fpr", "shd", "nnz", "precision", "f1score", "g_score"]},
                }
                pd.DataFrame([row]).to_csv(runtime_file, mode="a", header=False, index=False)

                max_in_degree_test = int(deg)
                max_out_degree_test = int(deg)
            except Exception as e:
                print(f"  Error at degree={deg}: {e}")
                continue

        pd.DataFrame(
            [
                {
                    "dataset": npz_path.stem,
                    "features_num": d,
                    "max_in_degree": max_in_degree_test,
                    "max_out_degree": max_out_degree_test,
                }
            ]
        ).to_csv(deg_file, mode="a", header=False, index=False)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = Path(_env_str("EXDBN_DATA_DIR", "datasets/syntheticdata/dynamic_er_gauss"))
    out_dir = Path(_env_str("EXDBN_OUT_DIR", str(repo_root / "results_sythetic_tables")))

    if _env_str("EXDBN_GENERATE_DYNAMIC", "0") == "1":
        dynamic_out = Path(
            _env_str(
                "EXDBN_GEN_OUT_DIR",
                str(repo_root / "datasets/syntheticdata/dynamic_test_one"),
            )
        )
        generate_dynamic_datasets(dynamic_out)
        return

    run_exdbn_on_npz_dir(data_dir=data_dir, out_dir=out_dir)
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
