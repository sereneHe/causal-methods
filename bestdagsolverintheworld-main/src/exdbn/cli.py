# src/exdbn/cli.py
import typer
from pathlib import Path
from omegaconf import OmegaConf

from exdbn.data.generate import generate_problem, save_problem
from exdbn.data.load import load_problem_from_npz
from exdbn.run import run_exdbn
from exdbn.config import MilpConfig

app = typer.Typer(help="EXDBN CLI")


# -------------------------
# generate
# -------------------------
@app.command()
def generate(
    mode: str = typer.Argument(..., help="dynamic"),
    d: int = typer.Option(10, help="Number of variables"),
    n: int = typer.Option(2000, help="Number of samples"),
    p: int = typer.Option(2, help="Number of lags"),
    out: Path = typer.Option(..., help="Output .npz file"),
):
    """
    Generate synthetic EXDBN dataset.
    """
    cfg = OmegaConf.create(
        {
            "problem": {
                "name": mode,
                "number_of_variables": d,
                "number_of_samples": n,
                "p": p,
                "generator": "notears",
                "noise_scale": 1.0,
                "intra_edge_ratio": 0.5,
                "inter_edge_ratio": 0.5,
                "w_max_intra": 1.0,
                "w_min_intra": 0.01,
                "w_max_inter": 0.2,
                "w_min_inter": 0.01,
                "w_decay": 1.0,
                "graph_type_intra": "er",
                "graph_type_inter": "er",
            }
        }
    )

    typer.echo("Generating dataset...")
    problem = generate_problem(cfg)
    save_problem(problem, out)
    typer.echo(f"Saved dataset to {out}")


# -------------------------
# run
# -------------------------
@app.command()
def run(
    data: Path = typer.Argument(..., help="Path to .npz dataset"),
    max_degree: int = typer.Option(5, help="Max in/out degree"),
    param1: float = typer.Option(0.1, help="Parameter 1 for MilpConfig"),
    param2: float = typer.Option(0.1, help="Parameter 2 for MilpConfig"),
):
    """
    Run EXDBN on a dataset.
    """
    problem = load_problem_from_npz(data)

    # Create MilpConfig
    milp_cfg = MilpConfig(param1=param1, param2=param2)

    metrics, gap = run_exdbn(problem, max_degree, milp_cfg)

    typer.echo(f"Gap: {gap:.4f}")
    for k, v in metrics.items():
        typer.echo(f"{k}: {v}")
