from invoke import task

@task
def exdbn(ctx, mode="run"):
    """
    Run EXDBN pipeline.
    """
    if mode == "generate":
        ctx.run("uv run exdbn generate", echo=True)
    else:
        ctx.run("uv run exdbn run", echo=True)

@task
def git(ctx, message):
    ctx.run("git add .", echo=True)
    ctx.run(f'git commit -m "{message}"', echo=True)
    ctx.run("git push", echo=True)

@task
def generate(ctx, mode="dynamic", d=10, n=2000, p=2, out="output.npz"):
    """
    Generate synthetic EXDBN dataset.
    """
    ctx.run(f"uv run exdbn generate {mode} --d={d} --n={n} --p={p} --out={out}", echo=True)
