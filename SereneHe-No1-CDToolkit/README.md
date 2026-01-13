# SereneHe-No1-CDToolkit

🎯 **A comprehensive causal discovery toolkit** combining algorithms from gCastle and bestdagsolverintheworld.

[![Algorithms](https://img.shields.io/badge/algorithms-20+-blue)]()
[![Methods](https://img.shields.io/badge/methods-complete-green)]()
[![Datasets](https://img.shields.io/badge/datasets-14-orange)]()

## ✨ Features

- **20+ Causal Discovery Algorithms** - Complete integration of gCastle and bestdagsolverintheworld
- **14 Datasets** - 7 synthetic + 7 real-world datasets
- **Comprehensive Evaluation** - Automated metrics, visualization, and comparison
- **Unified Interface** - Single API for all algorithms and datasets
- **Complete Documentation** - Examples, tutorials, and API docs

## 📋 Quick Links

- [Complete Verification Report](COMPLETE_VERIFICATION.md) - 100% coverage verification
- [Dataset Documentation](datasets/README.md) - All available datasets
- [Evaluation Examples](examples_evaluation.py) - How to use evaluation functions

## Structure

### Core Components

- **gcastle/** - Complete gCastle library with all causal discovery algorithms
  - `algorithms/gradient/` - Gradient-based methods (NOTEARS, GOLEM, DAG-GNN, GAE, GraN-DAG, MCSL, PNL, RL, CORL)
  - `algorithms/pc/` - PC algorithm
  - `algorithms/ges/` - GES algorithm
  - `algorithms/anm/` - ANM (Additive Noise Model)
  - `algorithms/lingam/` - LiNGAM algorithms (Direct, ICA)
  - `algorithms/ttpm/` - TTPM algorithm
  - `common/` - Common utilities
  - `datasets/` - Dataset utilities
  - `metrics/` - Evaluation metrics

### Exact Methods

- **exdbn/** - ExDBN (Exact Dynamic Bayesian Network) implementations
  - `solve_milp.py` - MILP-based solver for dynamic networks
  - `solve_lingam.py` - LiNGAM solver for DBN
  
- **exmag/** - ExMAG (Exact Maximal Ancestral Graph) solver implementations
  - `solve_exmag.py` - Primary ExMAG solver
  - `solve_exmag_2.py` - Alternative ExMAG implementation
  
- **exdag/** - ExDAG (Exact Directed Acyclic Graph) methods
  - `solve_boss.py` - BOSS solver
  - `solve_dagma.py` - DAGMA solver
  - `solve_nts_notears.py` - NOTEARS solver

### Additional Resources

- **examples/** - Example scripts for all gCastle algorithms
  - ANM, CORL, DAG-GNN, GAE, GES, GraN-DAG, LiNGAM, MCSL, NOTEARS, PC, PNL, RL, TTPM
- **scripts/** - Experiment scripts for running different algorithms
- **configs/** - Configuration files for solvers
- **results/** - Result analysis and visualization scripts
- **notebooks/** - Jupyter notebooks with examples and demonstrations
- **\*_utils.py** - 15+ utility modules for various tasks

## Included Algorithms

### From gCastle:
- **Gradient-based**: NOTEARS (Linear, Nonlinear, Low-rank), GOLEM, DAG-GNN, GAE, GraN-DAG, MCSL, PNL, RL, CORL
- **Score-based**: PC, GES
- **Function-based**: ANM, Direct-LiNGAM, ICA-LiNGAM
- **Causal functional model**: TTPM

### From bestdagsolverintheworld:
- **Exact methods**: ExDBN, ExMAG, ExDAG
- **Solvers**: MILP, BOSS, DAGMA, LiNGAM variants

## Sources

This toolkit integrates code from:
- [bestdagsolverintheworld](https://gitlab.fel.cvut.cz/rytirpav/bestdagsolverintheworld)
- [gCastle (trustworthyAI)](https://github.com/huawei-noah/trustworthyAI)

## Quick Start

### Using gCastle algorithms:
```python
from gcastle.algorithms import PC, NOTEARS, DirectLiNGAM
# See examples/ directory for detailed usage
```

### Using Exact methods:
See the notebooks directory for examples:
- `ExMAG_Berkeley_Admission_Example.ipynb`

## Installation

Install dependencies:
```bash
pip install -r requirements-gcastle.txt  # For gCastle
pip install -r requirements.txt          # For exact methods
pip install -r requirements-dagma.txt    # For DAGMA
pip install -r requirements-lingam.txt   # For LiNGAM
```

## Experiments

Run experiments using the scripts in the `scripts/` directory:
- `experiments_exdbn.sh` - ExDBN experiments
- `experiments_exmag.sh` - ExMAG experiments  
- `experiments_exdag.sh` - ExDAG experiments
