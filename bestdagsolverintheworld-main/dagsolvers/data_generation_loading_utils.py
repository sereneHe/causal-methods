import random
from os.path import join
import networkx as nx
import numpy as np
from omegaconf import DictConfig
from typing import Any, Dict

from dagsolvers.dagsolver_utils import ExDagDataException
from notears import utils


class _NullRun:
    """Minimal stand-in for the tracking run used in experiments.

    `load_problem` logs metadata and errors via `run`. For programmatic dataset
    generation we want the same codepaths without requiring MLflow/Tracking.
    """

    def log_param(self, *_args: Any, **_kwargs: Any) -> None:
        return

    def log_metric(self, *_args: Any, **_kwargs: Any) -> None:
        return

    def log_text(self, *_args: Any, **_kwargs: Any) -> None:
        return

def normalize_data(X, Y):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X = X - mean
    X = X / std

    for i, _ in enumerate(Y):
        Y[i] = Y[i] - mean
        Y[i] = Y[i] / std

    return X, Y

def load_problem(cfg: DictConfig, run):
    graph_type = cfg.problem.name
    #run.log_param('problem', graph_type)
    tabu_edges = []
    intra_nodes = None
    inter_nodes = None #TODO: all problems should define this
    if graph_type == 'cds':
        from dagsolvers import cds_utils
        W_true, B_true, A_true, B_lags_true, X, Y, intra_nodes, inter_nodes, tabu_edges = cds_utils.load_data(cfg.problem.n, cfg.problem.granularity, cfg.problem.p, cfg.problem.data_path)
        W_bi_true = np.zeros_like(W_true)
        B_bi_true = np.zeros_like(B_true)
    elif graph_type == 'codiet':
        from dagsolvers import codiet_utils
        W_true, B_true, A_true, B_lags_true, X, Y, intra_nodes, inter_nodes, tabu_edges = codiet_utils.load_data(cfg.problem)
        W_bi_true = np.zeros_like(W_true)
        B_bi_true = np.zeros_like(B_true)
    elif graph_type == 'krebs':
        p = cfg.problem.p
        measurements = cfg.problem.measurements
        variant = cfg.problem.variant
        #run.log_param('measurements', measurements)
        #run.log_param('variant', variant)
        from dagsolvers import krebs_utils
        W_true, B_true, A_true, B_lags_true, X, Y, X_lag, intra_nodes, inter_nodes = krebs_utils.load_data(variant, measurements, p, cfg.problem.data_path)
        W_bi_true = None
        B_bi_true = None
        #n, d = X.shape

    elif graph_type == 'Sachs':
        variant = cfg.problem.variant
        #run.log_param('variant', variant)
        #normalize = cfg.problem.get('normalize')
        #run.log_param('normalize', normalize)
        from dagsolvers import sachs_utils
        X, B_true, W_true = sachs_utils.load_data(variant, False, cfg.problem.data_path)
        # n, d = X.shape
        # p = 0
        Y = []
        A_true = []
        B_lags_true = []
        W_bi_true = None
        B_bi_true = None

    elif graph_type == 'bnlearn' or graph_type == 'nips2023':
        #run.log_param('variant', cfg.problem.variant)
        if graph_type == 'bnlearn':
            n = cfg.problem.get('number_of_samples')
            # normalize = cfg.problem.get('normalize', False)
            # run.log_param('normalize', normalize)
            from dagsolvers import bnlearn_utils
            X, W_true = bnlearn_utils.load_dataset(cfg.problem.variant, n=n, normalize=False)
        elif graph_type == 'nips2023':
            from dagsolvers import nips_comp_utils
            X, W_true = nips_comp_utils.load_dataset(cfg.problem.variant, cfg.problem.data_path)
        else:
            assert False
        B_true = W_true
        # n, d = X.shape
        # p = 0
        Y = []
        A_true = []
        B_lags_true = []
        W_bi_true = None
        B_bi_true = None

    elif graph_type == 'dynamic':
        #run.log_param('variant', cfg.problem.variant)
        p = cfg.problem.p
        d = cfg.problem.number_of_variables
        n = cfg.problem.number_of_samples
        degree_intra = cfg.problem.intra_edge_ratio * 2
        #run.log_param('intra_edge_ratio', cfg.problem.intra_edge_ratio)
        degree_inter = cfg.problem.inter_edge_ratio * 2
        #run.log_param('inter_edge_ratio', cfg.problem.inter_edge_ratio)
        w_max_inter = cfg.problem.w_max_inter
        #run.log_param('w_max_inter', w_max_inter)
        w_min_inter = cfg.problem.w_min_inter
        #run.log_param('w_min_inter', w_min_inter)
        w_decay = cfg.problem.w_decay
        #run.log_param('w_decay', w_decay)
        if p == 0:
            degree_inter = 0
        variant = cfg.problem.graph_type_intra
        if variant == 'er':
            graph_type_intra = 'erdos-renyi'
        elif variant == 'sf':
            graph_type_intra = 'barabasi-albert'
        else:
            assert False

        graph_type_inter = cfg.problem.graph_type_inter
        if graph_type_inter == 'er':
            graph_type_inter = 'erdos-renyi'

        #from structure.data_generators.wrappers import DataGenerationException
        try:
            generator = cfg.problem.generator
            noise_scale = cfg.problem.noise_scale
            noise_scale_variance = cfg.problem.get('noise_scale_variance', None)
            if noise_scale_variance is not None:
                noise_scale_vector = [random.uniform(noise_scale - noise_scale_variance, noise_scale + noise_scale_variance) for _ in range(d)]
            else:
                noise_scale_vector = [noise_scale] * d
            from structure.data_generators import gen_stationary_dyn_net_and_df
            g,df, intra_nodes, inter_nodes = gen_stationary_dyn_net_and_df(num_nodes=d, n_samples=n, p=p,
                                                                           degree_intra=degree_intra, degree_inter=degree_inter,
                                                                           graph_type_intra=graph_type_intra, graph_type_inter=graph_type_inter,
                                                                           w_max_intra=cfg.problem.w_max_intra, w_min_intra=cfg.problem.w_min_intra, w_min_inter=w_min_inter, w_max_inter=w_max_inter,
                                                                           w_decay=w_decay, noise_scale=noise_scale_vector, max_data_gen_trials=1000,
                                                                           generator=generator) #, w_min_inter=0.01, w_max_inter=0.2)
        except Exception as e: # DataGenerationException as e:
            run.log_text(f'Error: Cannot generate samples data. Exception: {e}', 'error.txt')
            run.log_metric('infeasible', True)
            raise ExDagDataException(e)

        W_true = nx.to_numpy_array(g, nodelist=intra_nodes)
        B_true = (W_true != 0).astype(int)
        a_mat = nx.to_numpy_array(g, nodelist=intra_nodes + inter_nodes)[len(intra_nodes) :, : len(intra_nodes)]
        df_x = df[intra_nodes]
        df_x_lag = df[inter_nodes]
        X = df_x.to_numpy()
        W_bi_true = None
        B_bi_true = None
        # s0 = degree_intra / 2 * d
        # B_true = utils.simulate_dag(d, s0, 'SF')
        # W_true = utils.simulate_parameter(B_true)
        # X = utils.simulate_linear_sem(W_true, n, 'gauss', noise_scale=1.0)
        # X_lag = df_x_lag.to_numpy()
        #X2 = utils.simulate_linear_sem(W_true, n, 'gauss', noise_scale=1.0)

        Y = []
        A_true = []
        B_lags_true = []
        for lag in range(1, p + 1):
            lag_cols = [c for c in inter_nodes if f'_lag{lag}' in c]
            df_x_lag = df[lag_cols]
            Y_lag = df_x_lag.to_numpy()
            Y.append(Y_lag)

            idxs = [f'_lag{lag}' in c for c in inter_nodes]
            a_mat_lag = a_mat[idxs,:]
            b_mat_lag = (a_mat_lag != 0).astype(int)
            A_true.append(a_mat_lag)
            B_lags_true.append(b_mat_lag)



    elif graph_type == 'er' or graph_type == 'sf' or graph_type == 'PATH' or graph_type == 'PATHPERM' or graph_type == 'G2':
        d = cfg.problem.number_of_variables
        n = cfg.problem.number_of_samples
        edge_ratio = cfg.problem.get('edge_ratio')
        s0 = None
        if edge_ratio is not None:
            s0 = int(round(edge_ratio * d))
        sem_type = cfg.problem.sem_type
        noise_scale = cfg.problem.noise_scale
        noise_scale_variance = cfg.problem.get('noise_scale_variance', None)
        if noise_scale_variance is not None:
            noise_scale = [random.uniform(noise_scale - noise_scale_variance, noise_scale + noise_scale_variance) for _ in range(d)]
        try:
            B_true = utils.simulate_dag(d, s0, graph_type.upper())
        except Exception as e:
            run.log_text(f'Error: Cannot generate samples data. Exception: {e}', 'error.txt')
            raise ExDagDataException(e)

        if cfg.problem.get('only_01', False):
            W_true = B_true
        elif cfg.problem.get('only_positive', False):
            W_true = utils.simulate_parameter(B_true, w_ranges=((0.5, 2.0),))
        else:
            W_true = utils.simulate_parameter(B_true)
        X = utils.simulate_linear_sem(W_true, n, sem_type, noise_scale=noise_scale, internal_normalization=cfg.problem.get('internal_normalization', False))
        Y = []
        A_true = []
        B_lags_true = []
        W_bi_true = None
        B_bi_true = None
        #p = 0

    elif graph_type == 'ermag':
        d = cfg.problem.number_of_variables
        n = cfg.problem.number_of_samples
        edge_ratio = cfg.problem.get('edge_ratio', 1)
        run.log_param('edge_ratio', edge_ratio)
        s0 = int(round(edge_ratio * d))
        sem_type = cfg.problem.sem_type
        noise_scale = cfg.problem.get('noise_scale', 1.0)
        tabu_edges_ratio = cfg.problem.get('tabu_edges_ratio', 0.2)
        hidden_vertices_ratio = cfg.problem.get('hidden_vertices_ratio', 0.2)
        run.log_param('sem_type', sem_type)
        run.log_param('noise_scale', noise_scale)
        try:
            B_true = utils.simulate_dag(d, s0, "ER")
        except Exception as e:
            run.log_text(f'Error: Cannot generate samples data. Exception: {e}', 'error.txt')
            run.log_metric('infeasible', True)
            raise ExDagDataException(e)

        if cfg.problem.get('only_01', False):
            W_true = B_true
        elif cfg.problem.get('only_positive', False):
            W_true = utils.simulate_parameter(B_true, w_ranges=((0.5, 2.0),))
        else:
            W_true = utils.simulate_parameter(B_true)
        X = utils.simulate_linear_sem(W_true, n, sem_type, noise_scale=noise_scale)
        Y = []
        A_true = []
        B_lags_true = []


        # p = 0

        new_d = int(d * (1-hidden_vertices_ratio))
        indices = np.random.choice(range(d), size=new_d, replace=False)
        d = new_d
        X = X[:, indices]
        W_true = W_true[np.ix_(indices, indices)]
        B_true = B_true[np.ix_(indices, indices)]
        B_bi_true = np.zeros_like(B_true)
        for i in range(d):
            for j in range(i):
                if W_true[i, j] == 0.0 and W_true[j, i] == 0.0:
                    if np.random.rand() < tabu_edges_ratio:
                        tabu_edges.append((i, j))
                        B_bi_true[i, j] = 1.0
                        B_bi_true[j, i] = 1.0 # Maybe dubious.

        W_bi_true = np.copy(B_bi_true)

    elif graph_type == 'bowfree_admg':
        from dagsolvers import admg_generators
        B_true, B_bi_true, tabu_edges, X = admg_generators.generate_graph_and_samples(cfg.problem.number_of_variables,cfg.problem.pdir, cfg.problem.pbidir, cfg.problem.max_in_arrows, cfg.problem.number_of_samples)
        W_true = B_true
        W_bi_true = B_bi_true
        Y = []
        A_true = []
        B_lags_true = []
    
    elif graph_type == 'admissions':
        import pandas as pd
        from sklearn.preprocessing import LabelEncoder
        file_path = cfg.problem.data_path
        df = pd.read_csv(file_path, sep=';')
        encoder = LabelEncoder()
        for col in ['gender', 'admit']:
            df[col] = encoder.fit_transform(df[col])
        #del df['gender']
        #df['gender'] = df['gender']
        #df['admit'] = 2 * df['admit']
        df = pd.get_dummies(df, columns=['department'], prefix='D', dtype=int)
        # for col in df.columns:
        #     if 'department' in col:
        #         df[col] = df[col] * 0.5
        # print(df.to_string())

        X = df.to_numpy()
        intra_nodes = df.columns.tolist()

        W_true = np.zeros((X.shape[1], X.shape[1]))
        B_true = np.zeros((X.shape[1], X.shape[1]))
        A_true = []
        B_lags_true = []
        Y = []
        W_bi_true = np.zeros((X.shape[1], X.shape[1]))
        B_bi_true = np.zeros((X.shape[1], X.shape[1]))
        tabu_edges = [(0,1),(1,0)]
    
    elif graph_type == 'load_from_file':
        assert False, 'implement me'

    else:
        assert False, 'unknown problem'
        
    # Generating default node names.
    if intra_nodes is None:
        intra_nodes = [f'node_{i}' for i in range(len(X[0]))]
    if inter_nodes is None:
        inter_nodes = [f'node_{i}_lag_{lag}' for lag in range(1, cfg.problem.get('p', 0) + 1) for i in range(len(X[0]))]
    
    return W_true, W_bi_true, B_true, B_bi_true, A_true, B_lags_true, X, Y, tabu_edges, intra_nodes, inter_nodes


def load_problem_dict(cfg: DictConfig) -> Dict[str, Any]:
    """Like `load_problem`, but returns a dict and doesn't require a tracking run.

    This is convenient for generating synthetic datasets programmatically.
    """

    (
        W_true,
        W_bi_true,
        B_true,
        B_bi_true,
        A_true,
        B_lags_true,
        X,
        Y,
        tabu_edges,
        intra_nodes,
        inter_nodes,
    ) = load_problem(cfg, _NullRun())

    return {
        "W_true": W_true,
        "W_bi_true": W_bi_true,
        "B_true": B_true,
        "B_bi_true": B_bi_true,
        "A_true": A_true,
        "B_lags_true": B_lags_true,
        "X": X,
        "Y": Y,
        "tabu_edges": tabu_edges,
        "intra_nodes": intra_nodes,
        "inter_nodes": inter_nodes,
    }

