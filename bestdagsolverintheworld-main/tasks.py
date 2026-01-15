# tasks.py
from invoke import task

@task
def generate_all(c):
    c.run("python cli.py generate all")

@task
def run_static(c):
    c.run("python cli.py run static --max-degrees 5 --sample-sizes 2000")
