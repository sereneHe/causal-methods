# generate.py
from pathlib import Path
import numpy as np
from omegaconf import OmegaConf

from dagsolvers.data_generation_loading_utils import load_problem_dict

def generate_datasets(out_dir: Path, is_dynamic: bool):
    features_list = [5,10,15,20,25,30,35]
    n = 2000

    if is_dynamic:
        problem_name = "dynamic"
        file_prefix = "dynamic"
    else:
        problem_name = "dynamic"
        file_prefix = "static"

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

        cfg = OmegaConf.create({"problem": problem_cfg})
        problem = load_problem_dict(cfg)

        np.savez(
            out_dir / f"{file_prefix}_er_gauss_{d}.npz",
            x=problem["X"],
            y=problem["W_true"],
        )

def generate_all_datasets(out_dir: Path):
    generate_datasets(out_dir / "static", is_dynamic=False)
    generate_datasets(out_dir / "dynamic", is_dynamic=True)
