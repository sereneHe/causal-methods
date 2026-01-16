# generate.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
from omegaconf import OmegaConf
from src.exdbn.data.generate_data import load_problem_dict

def generate_datasets_from_cfg(cfg: dict, out_dir: Path, is_dynamic: bool):
    features_list = cfg["generate"]["features_list"]
    n = cfg["generate"]["n_samples"]

    file_prefix = "dynamic" if is_dynamic else "static"
    out_dir.mkdir(parents=True, exist_ok=True)

    for d in features_list:
        print(f"[GEN] {file_prefix}: d={d}, n={n}")

        problem_cfg = {
            'name': 'dynamic',
            'generator': 'notears',
            'number_of_lags': 2 if is_dynamic else 0,
            'number_of_variables': d,
            'number_of_samples': n,
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

        cfg_omega = OmegaConf.create({"problem": problem_cfg})
        problem = load_problem_dict(cfg_omega)

        np.savez(
            out_dir / f"{file_prefix}_er_gauss_{d}.npz",
            x=problem["data"],
            y=problem.get("W_true", np.zeros((d, d)))  # or whatever your key is
        )

def generate_all_datasets_from_cfg(cfg: dict):
    generate_datasets_from_cfg(cfg, Path(cfg["data_dir"]) / "static", is_dynamic=False)
    generate_datasets_from_cfg(cfg, Path(cfg["data_dir"]) / "dynamic", is_dynamic=True)
