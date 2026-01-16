# cli.py
from __future__ import annotations
from pathlib import Path
import sys
import typer
import yaml

from exdbn.generate import generate_all_datasets_from_cfg, generate_datasets_from_cfg
from exdbn.run import run_exdbn_parallel_from_cfg

sys.path.append(str(Path(__file__).resolve().parents[2]))

app = typer.Typer(help="EXDBN CLI")
generate_app = typer.Typer(help="Generate synthetic datasets")
run_app = typer.Typer(help="Run EXDBN solver")

app.add_typer(generate_app, name="generate")
app.add_typer(run_app, name="run")

# -------------------------
# Helper to load config
# -------------------------
def load_config(config_path: Path):
    with open(config_path) as f:
        return yaml.safe_load(f)

# -------------------------
# Generate commands
# -------------------------
@generate_app.command("all")
def generate_all(
    config: Path = typer.Option(Path("config.yaml"), help="Path to config.yaml"),
):
    cfg = load_config(config)
    typer.echo(f"[INFO] Generating all datasets into {cfg['data_dir']}")
    generate_all_datasets_from_cfg(cfg)

@generate_app.command("single")
def generate_single(
    mode: str = typer.Option(..., help="static or dynamic"),
    config: Path = typer.Option(Path("config.yaml"), help="Path to config.yaml"),
):
    cfg = load_config(config)
    is_dynamic = mode.lower() == "dynamic"
    out_dir = Path(cfg["data_dir"]) / mode
    generate_datasets_from_cfg(cfg, out_dir=out_dir, is_dynamic=is_dynamic)

# -------------------------
# Run commands
# -------------------------
@run_app.command("static")
def run_static(
    config: Path = typer.Option(Path("config.yaml"), help="Path to config.yaml"),
    lambda1: float = typer.Option(None, help="Override lambda1"),
    lambda2: float = typer.Option(None, help="Override lambda2"),
    sample_sizes: list[int] = typer.Option(None, help="Override sample sizes"),
    max_degrees: list[int] = typer.Option(None, help="Override max degrees"),
    num_workers: int = typer.Option(None, help="Override number of workers"),
):
    cfg = load_config(config)
    run_exdbn_parallel_from_cfg(
        cfg, mode="static",
        lambda1=lambda1, lambda2=lambda2,
        sample_sizes=sample_sizes,
        max_degrees=max_degrees,
        num_workers=num_workers
    )

@run_app.command("dynamic")
def run_dynamic(
    config: Path = typer.Option(Path("config.yaml"), help="Path to config.yaml"),
    lambda1: float = typer.Option(None),
    lambda2: float = typer.Option(None),
    sample_sizes: list[int] = typer.Option(None),
    max_degrees: list[int] = typer.Option(None),
    num_workers: int = typer.Option(None),
):
    cfg = load_config(config)
    run_exdbn_parallel_from_cfg(
        cfg, mode="dynamic",
        lambda1=lambda1, lambda2=lambda2,
        sample_sizes=sample_sizes,
        max_degrees=max_degrees,
        num_workers=num_workers
    )

# -------------------------
# CLI entry
# -------------------------
if __name__ == "__main__":
    app()
