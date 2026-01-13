"""
Unified data loading interface for SereneHe-No1-CDToolkit
Supports both synthetic and real datasets for causal discovery

Synthetic Datasets:
- 'er', 'sf', 'PATH', 'PATHPERM', 'G2' - Static DAG structures
- 'ermag' - MAG with hidden variables and bidirected edges
- 'dynamic' - Dynamic Bayesian Networks with lags

Real Datasets:
- 'admissions' - Berkeley admissions data
- 'krebs' - Krebs cycle data
- 'codiet' - CoDiet dataset
- 'cds' - CDS dataset
- 'Sachs' - Sachs protein signaling data
- 'bnlearn' - Various bnlearn benchmark datasets
- 'nips2023' - NIPS 2023 competition datasets
"""

import random
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, Tuple, List, Optional


class ExDagDataException(Exception):
    """Exception raised for errors in data generation"""
    pass


def simulate_dag(d: int, s0: float, graph_type: str) -> np.ndarray:
    """
    Simulate random DAG with specified number of nodes and edges
    
    Args:
        d: number of nodes
        s0: expected number of edges
        graph_type: 'ER', 'SF', 'PATH', 'PATHPERM', 'G2'
    
    Returns:
        Binary adjacency matrix of DAG
    """
    def _random_permutation(M):
        P = np.random.permutation(np.eye(M.shape[0]))
        return P.T @ M @ P
    
    def _random_acyclic_orientation(B_und):
        return np.tril(_random_permutation(B_und), k=-1)
    
    def _graph_to_adjmat(G):
        return np.array(nx.to_numpy_array(G))
    
    if graph_type == 'ER':
        # Erdos-Renyi
        G_und = nx.erdos_renyi_graph(n=d, p=s0/(d*(d-1)/2))
        B_und = _graph_to_adjmat(G_und)
        B = _random_acyclic_orientation(B_und)
    elif graph_type == 'SF':
        # Scale-free, Barabasi-Albert
        m = int(round(s0 / d))
        G = nx.barabasi_albert_graph(n=d, m=m)
        B_und = _graph_to_adjmat(G)
        B = _random_acyclic_orientation(B_und)
    elif graph_type == 'PATH':
        # Path graph
        B = np.zeros([d, d])
        for i in range(d-1):
            B[i, i+1] = 1
    elif graph_type == 'PATHPERM':
        # Path graph with random permutation
        B = np.zeros([d, d])
        for i in range(d-1):
            B[i, i+1] = 1
        B = _random_permutation(B)
    elif graph_type == 'G2':
        # Two-component graph
        B = np.zeros([d, d])
        B[0, 1] = 1
        B[0, 2] = 1
    else:
        raise ValueError(f'Unknown graph type: {graph_type}')
    
    B_perm = _random_permutation(B)
    assert is_dag(B_perm)
    return B_perm


def is_dag(W: np.ndarray) -> bool:
    """Check if W is a DAG"""
    G = nx.DiGraph(W)
    return nx.is_directed_acyclic_graph(G)


def simulate_parameter(B: np.ndarray, w_ranges=((-2.0, -0.5), (0.5, 2.0))) -> np.ndarray:
    """
    Simulate parameters for the weighted adjacency matrix
    
    Args:
        B: binary adjacency matrix
        w_ranges: tuple of weight ranges for edge parameters
    
    Returns:
        Weighted adjacency matrix
    """
    W = np.zeros(B.shape)
    S = np.random.randint(len(w_ranges), size=B.shape)
    for i, (low, high) in enumerate(w_ranges):
        U = np.random.uniform(low=low, high=high, size=B.shape)
        W += B * (S == i) * U
    return W


def simulate_linear_sem(W: np.ndarray, n: int, sem_type: str, 
                       noise_scale: float = 1.0) -> np.ndarray:
    """
    Simulate samples from linear SEM with specified noise distribution
    
    Args:
        W: weighted adjacency matrix
        n: number of samples
        sem_type: noise type ('gauss', 'exp', 'gumbel', 'uniform')
        noise_scale: scale of noise distribution
    
    Returns:
        [n, d] sample matrix
    """
    d = W.shape[0]
    if sem_type == 'gauss':
        noise = np.random.normal(scale=noise_scale, size=(n, d))
    elif sem_type == 'exp':
        noise = np.random.exponential(scale=noise_scale, size=(n, d))
    elif sem_type == 'gumbel':
        noise = np.random.gumbel(scale=noise_scale, size=(n, d))
    elif sem_type == 'uniform':
        noise = np.random.uniform(low=-noise_scale, high=noise_scale, size=(n, d))
    else:
        raise ValueError(f'Unknown sem type: {sem_type}')
    
    G = nx.DiGraph(W)
    ordered_vertices = list(nx.topological_sort(G))
    assert len(ordered_vertices) == d
    X = np.zeros([n, d])
    for j in ordered_vertices:
        parents = list(G.predecessors(j))
        X[:, j] = X[:, parents] @ W[parents, j] + noise[:, j]
    
    return X


def load_problem_dict(problem: Dict) -> Tuple:
    """
    Load problem data based on problem configuration dictionary
    
    Args:
        problem: Configuration dictionary with keys:
            - 'name': problem type ('er', 'sf', 'ermag', 'dynamic', 'admissions', etc.)
            - 'number_of_variables': number of variables (for synthetic)
            - 'number_of_samples': number of samples
            - 'sem_type': noise type (default: 'gauss')
            - 'noise_scale': noise scale (default: 1.0)
            - 'edge_ratio': ratio of edges to nodes (for synthetic)
            - other problem-specific parameters
    
    Returns:
        Tuple of (W_true, W_bi_true, B_true, B_bi_true, A_true, X, Y, 
                 tabu_edges, intra_nodes, inter_nodes)
    """
    graph_type = problem['name']
    tabu_edges = []
    intra_nodes = None
    inter_nodes = None
    W_bi_true = None
    B_bi_true = None
    A_true = []
    Y = []
    
    # ========== Synthetic Static DAGs ==========
    if graph_type in ['er', 'sf', 'PATH', 'PATHPERM', 'G2']:
        d = problem['number_of_variables']
        n = problem['number_of_samples']
        edge_ratio = problem.get('edge_ratio', 1.0)
        s0 = edge_ratio * d
        sem_type = problem.get('sem_type', 'gauss')
        noise_scale = problem.get('noise_scale', 1.0)
        
        B_true = simulate_dag(d, s0, graph_type.upper())
        W_true = simulate_parameter(B_true)
        X = simulate_linear_sem(W_true, n, sem_type, noise_scale=noise_scale)
    
    # ========== MAG with Hidden Variables ==========
    elif graph_type == 'ermag':
        d = problem['number_of_variables']
        n = problem['number_of_samples']
        edge_ratio = problem.get('edge_ratio', 1.0)
        s0 = edge_ratio * d
        sem_type = problem.get('sem_type', 'gauss')
        noise_scale = problem.get('noise_scale', 1.0)
        tabu_edges_ratio = problem.get('tabu_edges_ratio', 0.2)
        hidden_vertices_ratio = problem.get('hidden_vertices_ratio', 0.2)
        
        try:
            B_true = simulate_dag(d, s0, "ER")
        except Exception as e:
            raise ExDagDataException(f"Failed to generate DAG: {e}")
        
        if problem.get('only_01', False):
            W_true = B_true
        elif problem.get('only_positive', False):
            W_true = simulate_parameter(B_true, w_ranges=((0.5, 2.0),))
        else:
            W_true = simulate_parameter(B_true)
        
        X = simulate_linear_sem(W_true, n, sem_type, noise_scale=noise_scale)
        Y = []
        A_true = []
        
        # Remove hidden vertices
        new_d = int(d * (1 - hidden_vertices_ratio))
        indices = np.random.choice(range(d), size=new_d, replace=False)
        d = new_d
        X = X[:, indices]
        W_true = W_true[np.ix_(indices, indices)]
        B_true = B_true[np.ix_(indices, indices)]
        
        # Add bidirected edges (tabu edges)
        B_bi_true = np.zeros_like(B_true)
        for i in range(d):
            for j in range(i):
                if W_true[i, j] == 0.0 and W_true[j, i] == 0.0:
                    if np.random.rand() < tabu_edges_ratio:
                        tabu_edges.append((i, j))
                        B_bi_true[i, j] = 1.0
                        B_bi_true[j, i] = 1.0
        W_bi_true = np.copy(B_bi_true)
    
    # ========== Dynamic Bayesian Networks ==========
    elif graph_type == 'dynamic':
        p = problem.get('number_of_lags', 1)
        d = problem['number_of_variables']
        n = problem['number_of_samples']
        intra_edge_ratio = problem.get('intra_edge_ratio', 1.0)
        degree_intra = intra_edge_ratio * 2
        inter_edge_ratio = problem.get('inter_edge_ratio', 1.0)
        degree_inter = inter_edge_ratio * 2
        w_max_inter = problem.get('w_max_inter', 0.2)
        w_min_inter = problem.get('w_min_inter', 0.01)
        w_decay = problem.get('w_decay', 0.1)
        
        if p == 0:
            degree_inter = 0
        
        variant = problem.get('graph_type_intra', 'er')
        if variant == 'er':
            graph_type_intra = 'erdos-renyi'
        elif variant == 'sf':
            graph_type_intra = 'barabasi-albert'
        else:
            raise ValueError(f'Unknown graph type: {variant}')
        
        graph_type_inter = problem.get('graph_type_inter', 'er')
        if graph_type_inter == 'er':
            graph_type_inter = 'erdos-renyi'
        
        try:
            generator = problem.get('generator', 'ts')
            noise_scale = problem.get('noise_scale', 1.0)
            noise_scale_variance = problem.get('noise_scale_variance', None)
            
            if noise_scale_variance is not None:
                noise_scale_vector = [
                    random.uniform(noise_scale - noise_scale_variance, 
                                 noise_scale + noise_scale_variance) 
                    for _ in range(d)
                ]
            else:
                noise_scale_vector = [noise_scale] * d
            
            # Try to import from structure module
            try:
                from structure.data_generators import gen_stationary_dyn_net_and_df
            except ImportError:
                print("Warning: gen_stationary_dyn_net_and_df not found. Using mock implementation.")
                # Mock implementation
                def gen_stationary_dyn_net_and_df(num_nodes, n_samples, p, degree_intra, degree_inter,
                                                 graph_type_intra, graph_type_inter, w_max_intra, w_min_intra,
                                                 w_min_inter, w_max_inter, w_decay, noise_scale, max_data_gen_trials,
                                                 generator):
                    g = nx.DiGraph()
                    intra_nodes_mock = [f'node_{i}' for i in range(num_nodes)]
                    g.add_nodes_from(intra_nodes_mock)
                    inter_nodes_mock = []
                    for lag in range(1, p + 1):
                        inter_nodes_mock.extend([f'node_{i}_lag{lag}' for i in range(num_nodes)])
                    g.add_nodes_from(inter_nodes_mock)
                    df_mock_data = np.random.rand(n_samples, num_nodes + len(inter_nodes_mock))
                    df_mock_cols = intra_nodes_mock + inter_nodes_mock
                    df_mock = pd.DataFrame(df_mock_data, columns=df_mock_cols)
                    return g, df_mock, intra_nodes_mock, inter_nodes_mock
            
            g, df, intra_nodes, inter_nodes = gen_stationary_dyn_net_and_df(
                num_nodes=d, n_samples=n, p=p,
                degree_intra=degree_intra, degree_inter=degree_inter,
                graph_type_intra=graph_type_intra, graph_type_inter=graph_type_inter,
                w_max_intra=problem.get('w_max_intra', 0.5),
                w_min_intra=problem.get('w_min_intra', 0.01),
                w_min_inter=w_min_inter, w_max_inter=w_max_inter,
                w_decay=w_decay, noise_scale=noise_scale_vector, 
                max_data_gen_trials=1000, generator=generator
            )
            df = df.copy()
        except Exception as e:
            raise ExDagDataException(f"Failed to generate dynamic network: {e}")
        
        W_true = nx.to_numpy_array(g, nodelist=intra_nodes)
        B_true = W_true != 0
        a_mat = nx.to_numpy_array(g, nodelist=intra_nodes + inter_nodes)[len(intra_nodes):, :len(intra_nodes)]
        df_x = df[intra_nodes]
        X = df_x.to_numpy()
        W_bi_true = None
        B_bi_true = None
        
        Y = []
        A_true = []
        for lag in range(1, p + 1):
            lag_cols = [c for c in inter_nodes if f'_lag{lag}' in c]
            df_x_lag = df[lag_cols]
            Y_lag = df_x_lag.to_numpy()
            Y.append(Y_lag)
            
            idxs = [f'_lag{lag}' in c for c in inter_nodes]
            a_mat_lag = a_mat[idxs, :]
            A_true.append(a_mat_lag)
    
    # ========== Real Datasets ==========
    elif graph_type == 'admissions':
        # Berkeley Admissions dataset
        file_path = 'https://raw.githubusercontent.com/sereneHe/my-data/refs/heads/main/UCBadmit_long_samples.csv'
        df = pd.read_csv(file_path, sep=';')
        n = problem.get('number_of_samples', len(df))
        df = df.sample(n=min(n, len(df)), random_state=42)
        X = df.to_numpy()
        intra_nodes = df.columns.tolist()
        W_true = np.array([[0, 0, 1],
                          [1, 0, 0],
                          [0, 0, 0]])
        B_true = np.zeros((X.shape[1], X.shape[1]))
        W_bi_true = np.zeros_like(W_true)
        B_bi_true = np.zeros_like(B_true)
        tabu_edges = [(0, 1), (1, 0)]
    
    elif graph_type == 'krebs':
        # Krebs cycle data - requires krebs_utils
        try:
            import sys
            import os
            sys.path.insert(0, os.path.dirname(__file__))
            from krebs_utils import load_data as load_krebs
            p = problem.get('number_of_lags', 1)
            measurements = problem.get('measurements', 'all')
            variant = problem.get('variant', 'default')
            data_path = problem.get('data_path', '')
            W_true, B_true, A_true, B_lags_true, X, Y, X_lag, intra_nodes, inter_nodes = load_krebs(
                variant, measurements, p, data_path
            )
            W_bi_true = None
            B_bi_true = None
        except ImportError:
            raise ExDagDataException("krebs_utils not available")
    
    elif graph_type == 'codiet':
        # CoDiet dataset
        try:
            import sys
            import os
            sys.path.insert(0, os.path.dirname(__file__))
            from codiet_utils import load_data as load_codiet
            W_true, B_true, A_true, B_lags_true, X, Y, intra_nodes, inter_nodes, tabu_edges = load_codiet(problem)
            W_bi_true = np.zeros_like(W_true)
            B_bi_true = np.zeros_like(B_true)
        except ImportError:
            raise ExDagDataException("codiet_utils not available")
    
    elif graph_type == 'cds':
        # CDS dataset
        try:
            import sys
            import os
            sys.path.insert(0, os.path.dirname(__file__))
            from cds_utils import load_data as load_cds
            n = problem.get('number_of_samples', 100)
            granularity = problem.get('granularity', 'default')
            p = problem.get('number_of_lags', 1)
            data_path = problem.get('data_path', '')
            W_true, B_true, A_true, B_lags_true, X, Y, intra_nodes, inter_nodes, tabu_edges = load_cds(
                n, granularity, p, data_path
            )
            W_bi_true = np.zeros_like(W_true)
            B_bi_true = np.zeros_like(B_true)
        except ImportError:
            raise ExDagDataException("cds_utils not available")
    
    elif graph_type == 'Sachs':
        # Sachs protein signaling data
        try:
            import sys
            import os
            sys.path.insert(0, os.path.dirname(__file__))
            from sachs_utils import load_data as load_sachs
            variant = problem.get('variant', 'default')
            data_path = problem.get('data_path', '')
            X, B_true, W_true = load_sachs(variant, False, data_path)
            Y = []
            A_true = []
            W_bi_true = None
            B_bi_true = None
        except ImportError:
            raise ExDagDataException("sachs_utils not available")
    
    elif graph_type == 'bnlearn':
        # BNLearn benchmark datasets
        try:
            import sys
            import os
            sys.path.insert(0, os.path.dirname(__file__))
            from bnlearn_utils import load_dataset as load_bnlearn
            n = problem.get('number_of_samples', 1000)
            variant = problem.get('variant', 'asia')
            X, W_true = load_bnlearn(variant, n=n, normalize=False)
            B_true = W_true
            Y = []
            A_true = []
            W_bi_true = None
            B_bi_true = None
        except ImportError:
            raise ExDagDataException("bnlearn_utils not available")
    
    elif graph_type == 'nips2023':
        # NIPS 2023 competition datasets
        try:
            import sys
            import os
            sys.path.insert(0, os.path.dirname(__file__))
            from nips_comp_utils import load_dataset as load_nips
            variant = problem.get('variant', 'default')
            data_path = problem.get('data_path', '')
            X, W_true = load_nips(variant, data_path)
            B_true = W_true
            Y = []
            A_true = []
            W_bi_true = None
            B_bi_true = None
        except ImportError:
            raise ExDagDataException("nips_comp_utils not available")
    
    else:
        raise ValueError(f"Unknown problem type: {graph_type}")
    
    # Set default node names if not specified
    if intra_nodes is None:
        intra_nodes = [f'node_{i}' for i in range(X.shape[1])]
    if inter_nodes is None:
        inter_nodes = []
    
    return W_true, W_bi_true, B_true, B_bi_true, A_true, X, Y, tabu_edges, intra_nodes, inter_nodes


# Convenience function for backward compatibility
def load_problem(problem: Dict) -> Tuple:
    """Alias for load_problem_dict"""
    return load_problem_dict(problem)
