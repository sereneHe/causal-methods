# run_all.py
from pathlib import Path
import sys
import typer
import yaml

from generate import generate_all_datasets_from_cfg
from run import run_exdbn_parallel_from_cfg

sys.path.append(str(Path(__file__).resolve().parents[2]))

app = typer.Typer(help="EXDBN Quick Launch Script")

def load_config(config_path: Path):
    """Load YAML configuration file"""
    with open(config_path) as f:
        return yaml.safe_load(f)

@app.command()
def run_all(
    config: Path = typer.Option(Path("config.yaml"), help="Path to config.yaml"),
    lambda1: float = typer.Option(None, help="Override lambda1 regularization coefficient"),
    lambda2: float = typer.Option(None, help="Override lambda2 regularization coefficient"),
    sample_sizes: list[int] = typer.Option(None, help="Override sample sizes"),
    max_degrees: list[int] = typer.Option(None, help="Override maximum degrees"),
    num_workers: int = typer.Option(None, help="Override number of parallel workers"),
):
    """
    Run the full EXDBN pipeline in one command:
    1️⃣ Generate synthetic datasets (static + dynamic)
    2️⃣ Run EXDBN on static datasets
    3️⃣ Run EXDBN on dynamic datasets
    """

    cfg = load_config(config)

    # -----------------------
    # Generate datasets
    # -----------------------
    typer.echo("[INFO] Generating datasets...")
    generate_all_datasets_from_cfg(cfg)

    # -----------------------
    # Run EXDBN on static datasets
    # -----------------------
    typer.echo("[INFO] Running EXDBN on static datasets...")
    run_exdbn_parallel_from_cfg(cfg, mode="static",
                                lambda1=lambda1,
                                lambda2=lambda2,
                                sample_sizes=sample_sizes,
                                max_degrees=max_degrees,
                                num_workers=num_workers)

    # -----------------------
    # Run EXDBN on dynamic datasets
    # -----------------------
    typer.echo("[INFO] Running EXDBN on dynamic datasets...")
    run_exdbn_parallel_from_cfg(cfg, mode="dynamic",
                                lambda1=lambda1,
                                lambda2=lambda2,
                                sample_sizes=sample_sizes,
                                max_degrees=max_degrees,
                                num_workers=num_workers)

    typer.echo("[DONE] Full EXDBN pipeline completed!")

if __name__ == "__main__":
    app()
