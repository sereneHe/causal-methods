# cli.py
from __future__ import annotations
from pathlib import Path
import sys

import typer

from exdbn.generate import generate_all_datasets, generate_datasets
from exdbn.run import run_exdbn_parallel

sys.path.append(str(Path(__file__).resolve().parent.parent))

app = typer.Typer(help="EXDBN CLI")
generate_app = typer.Typer(help="Generate synthetic datasets")
run_app = typer.Typer(help="Run EXDBN solver")

app.add_typer(generate_app, name="generate")
app.add_typer(run_app, name="run")

# -------------------------
# Generate commands
# -------------------------
@generate_app.command("all")
def generate_all(out: Path = typer.Option(Path("datasets/syntheticdata"), help="Output base directory")):
    typer.echo(f"[INFO] Generating all datasets into {out}")
    generate_all_datasets(out_dir=out)

@generate_app.command("single")
def generate_single(
    out: Path = typer.Option(Path("datasets/syntheticdata"), help="Output directory"),
    mode: str = typer.Option(..., help="static or dynamic")
):
    is_dynamic = mode.lower() == "dynamic"
    generate_datasets(out_dir=out, is_dynamic=is_dynamic)

# -------------------------
# Run commands
# -------------------------
@run_app.command("static")
def run_static(
    data_dir: Path = typer.Option(Path("datasets/syntheticdata/static"), help="Directory with npz datasets"),
    out_dir: Path = typer.Option(Path("results/exdbn/static"), help="Output directory"),
    sample_sizes: list[int] = typer.Option([2000], help="List of sample sizes"),
    max_degrees: list[int] = typer.Option([5], help="List of max degrees"),
    lambda1: float = typer.Option(1.0),
    lambda2: float = typer.Option(1.0),
    num_workers: int = typer.Option(None, help="Number of parallel workers"),
):
    run_exdbn_parallel(
        base_data=data_dir,
        base_out=out_dir,
        sample_sizes=sample_sizes,
        max_degrees=max_degrees,
        lambda1=lambda1,
        lambda2=lambda2,
        mode="static",
        num_workers=num_workers,
    )

@run_app.command("dynamic")
def run_dynamic(
    data_dir: Path = typer.Option(Path("datasets/syntheticdata/dynamic")),
    out_dir: Path = typer.Option(Path("results/exdbn/dynamic")),
    sample_sizes: list[int] = typer.Option([2000]),
    max_degrees: list[int] = typer.Option([5]),
    lambda1: float = typer.Option(1.0),
    lambda2: float = typer.Option(1.0),
    num_workers: int = typer.Option(None),
):
    run_exdbn_parallel(
        base_data=data_dir,
        base_out=out_dir,
        sample_sizes=sample_sizes,
        max_degrees=max_degrees,
        lambda1=lambda1,
        lambda2=lambda2,
        mode="dynamic",
        num_workers=num_workers,
    )

# -------------------------
# CLI entry
# -------------------------
if __name__ == "__main__":
    app()
