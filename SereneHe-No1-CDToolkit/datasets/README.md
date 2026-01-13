# Datasets Module for SereneHe-No1-CDToolkit

This module provides unified data loading functions for both synthetic and real causal discovery datasets.

## Structure

- `load_problem.py` - Main data loading interface with `load_problem_dict()` function
- `gcastle/` - gCastle's dataset utilities (simulator, builtin datasets)
- `datasets_real/` - Real-world datasets (admissions, etc.)

## Supported Datasets

### Synthetic Datasets

#### Static DAG Structures
- **'er'** - Erdos-Renyi random graph
- **'sf'** - Scale-Free (Barabasi-Albert) graph
- **'PATH'** - Path graph
- **'PATHPERM'** - Path graph with random permutation
- **'G2'** - Two-component graph

#### MAG with Hidden Variables
- **'ermag'** - ER graph with hidden variables and bidirected edges

#### Dynamic Networks
- **'dynamic'** - Dynamic Bayesian Networks with time lags

### Real Datasets

- **'admissions'** - Berkeley admissions data (gender bias example)
- **'krebs'** - Krebs cycle biological data
- **'codiet'** - CoDiet dataset
- **'cds'** - CDS dataset
- **'Sachs'** - Sachs protein signaling network data
- **'bnlearn'** - BNLearn benchmark datasets (asia, sachs, alarm, etc.)
- **'nips2023'** - NIPS 2023 causal discovery competition datasets

## Usage

### Basic Example

```python
from datasets.load_problem import load_problem_dict

# Load synthetic ER graph
problem_config = {
    'name': 'er',
    'number_of_variables': 10,
    'number_of_samples': 1000,
    'edge_ratio': 2.0,
    'sem_type': 'gauss',
    'noise_scale': 1.0
}

W_true, W_bi_true, B_true, B_bi_true, A_true, X, Y, tabu_edges, intra_nodes, inter_nodes = \
    load_problem_dict(problem_config)
```

### Synthetic Data Examples

#### Erdos-Renyi DAG
```python
problem = {
    'name': 'er',
    'number_of_variables': 20,
    'number_of_samples': 500,
    'edge_ratio': 1.5,  # 1.5 * d edges
    'sem_type': 'gauss',  # or 'exp', 'gumbel', 'uniform'
    'noise_scale': 1.0
}
```

#### Scale-Free DAG
```python
problem = {
    'name': 'sf',
    'number_of_variables': 15,
    'number_of_samples': 1000,
    'edge_ratio': 2.0,
    'sem_type': 'exp',
    'noise_scale': 0.8
}
```

#### MAG with Hidden Variables
```python
problem = {
    'name': 'ermag',
    'number_of_variables': 10,
    'number_of_samples': 500,
    'edge_ratio': 1.5,
    'tabu_edges_ratio': 0.2,  # 20% of non-edges become bidirected
    'hidden_vertices_ratio': 0.3,  # 30% variables are hidden
    'sem_type': 'gauss'
}
```

#### Dynamic Bayesian Network
```python
problem = {
    'name': 'dynamic',
    'number_of_variables': 5,
    'number_of_samples': 1000,
    'number_of_lags': 2,  # 2 time lags
    'intra_edge_ratio': 2.0,  # edges within time slice
    'inter_edge_ratio': 1.0,  # edges between time slices
    'graph_type_intra': 'er',  # or 'sf'
    'w_max_inter': 0.4,
    'w_min_inter': 0.1,
    'noise_scale': 1.0
}
```

### Real Data Examples

#### Berkeley Admissions
```python
problem = {
    'name': 'admissions',
    'number_of_samples': 100
}
```

#### Sachs Protein Data
```python
problem = {
    'name': 'Sachs',
    'variant': 'observational',  # or 'interventional'
    'data_path': '/path/to/data'
}
```

#### BNLearn Datasets
```python
problem = {
    'name': 'bnlearn',
    'variant': 'asia',  # 'sachs', 'alarm', 'child', etc.
    'number_of_samples': 1000
}
```

## Return Values

All functions return a tuple of:

- **W_true**: True weighted adjacency matrix (DAG edges)
- **W_bi_true**: Weighted bidirected edges (for MAGs)
- **B_true**: Binary adjacency matrix
- **B_bi_true**: Binary bidirected edges
- **A_true**: List of lag matrices (for dynamic networks)
- **X**: Data matrix [n_samples, n_variables]
- **Y**: List of lagged data matrices (for dynamic networks)
- **tabu_edges**: List of forbidden edges
- **intra_nodes**: List of variable names
- **inter_nodes**: List of lagged variable names

## Parameters

### Common Parameters

- `name` (str): Dataset type
- `number_of_variables` (int): Number of variables (synthetic only)
- `number_of_samples` (int): Number of samples

### Synthetic Data Parameters

- `edge_ratio` (float): Ratio of edges to nodes
- `sem_type` (str): Noise distribution ('gauss', 'exp', 'gumbel', 'uniform')
- `noise_scale` (float): Standard deviation/scale of noise

### Dynamic Network Parameters

- `number_of_lags` (int): Number of time lags
- `intra_edge_ratio` (float): Edge ratio within time slices
- `inter_edge_ratio` (float): Edge ratio between time slices
- `graph_type_intra` (str): Intra-slice graph type ('er', 'sf')
- `w_max_inter`, `w_min_inter` (float): Weight ranges for inter-slice edges

### MAG Parameters

- `tabu_edges_ratio` (float): Proportion of non-edges to make bidirected
- `hidden_vertices_ratio` (float): Proportion of variables to hide

### Real Data Parameters

- `variant` (str): Specific variant of the dataset
- `data_path` (str): Path to data files
