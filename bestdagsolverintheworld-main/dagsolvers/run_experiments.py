import time
import logging
from os.path import join
import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from mlflow import log_metric, log_text

from dagsolvers.ploting_utils import make_plots
from dagsolvers.solve_causal_learn import solve_fci

from dagsolvers.solvers.IPBMandHforCGL.solver_ipbm import solve_ipbm
from dagsolvers.sortnregress import sortnregress
from dagsolvers.varsortability import varsortability

logger = logging.getLogger(__name__)

from omegaconf import DictConfig, OmegaConf

from dagsolvers.dagsolver_utils import plot, \
    find_optimal_multiple_thresholds, ExDagDataException, plot_heatmap
from dagsolvers.data_generation_loading_utils import load_problem, normalize_data
from dagsolvers.experiments_utils import log_params_from_omegaconf_dict, log_system_info
from dagsolvers.metrics_utils import calculate_metrics, calculate_metrics_pag, count_accuracy, compute_norm_distance, \
    least_square_cost, apply_threshold, find_optimal_threshold_for_shd
from dagsolvers.utils import log_exceptions
from notears import utils, linear
import solve_milp
import solve_exmag
import solve_gobnilp
from structure.dynotears import from_numpy_dynamic
import tracking_utils as tu


@hydra.main(version_base=None,  config_path="./experiments_conf", config_name="config")
@log_exceptions
def start_experiment(cfg: DictConfig) -> None:
    tu.set_tracking(cfg.tracking, cfg.experiment) #mlflow.set_experiment(experiment_name=cfg.experiment)
    conf_yaml = OmegaConf.to_yaml(cfg)
    print(conf_yaml)
    app_config = OmegaConf.to_container(cfg)
    seed = cfg.problem.seed


    utils.set_random_seed(seed)
    with (tu.start_run() as run):
        #log_params_from_omegaconf_dict(cfg)
        log_params_from_omegaconf_dict(cfg, {'solver', 'problem'})
        log_system_info(HydraConfig.get())
        note = cfg.get('note','')
        run.log_param('note', note)
        output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        run.log_text(output_dir, 'work_dir.txt')
        logger.info(f'Starting experiment in {output_dir}')
        with open(join(output_dir, 'config.yaml'), 'w') as f:
            f.write(conf_yaml)
        run.log_artifact(join(output_dir, 'config.yaml'))

        try:
            W_true, W_bi_true, B_true, B_bi_true, W_lags_true, B_lags_true, X, Y, tabu_edges, intra_nodes, inter_nodes = load_problem(cfg, run)
            if X.shape[0] > 1:
                varsort = varsortability(X, B_true)
            else:
                varsort = 0
            print(f'VARSORT: {varsort}')
            run.log_metric('varsortability', varsort)
            variances = np.var(X, axis=0)
            means = np.mean(X, axis=0)
            logger.info(f'DATA MEANS: {means}')
            logger.info(f'DATA VARIANCES: {variances}')
            if cfg.problem.normalize:
                #run.log_param('normalize', cfg.problem.normalize)
                X, Y = normalize_data(X,Y)
                print(f'varsort: {varsortability(X, B_true)}')
                # variances = np.var(X, axis=0)
                # means = np.mean(X, axis=0)
                logger.info(f'DATA NORMALIZED:')
                # print(means)
                # print(variances)
        except ExDagDataException as e:
            logging.critical(e, exc_info=True)
            return

        n, d = X.shape
        p = len(W_lags_true) if W_lags_true is not None else 0
        B_true = B_true.astype(int)
        if B_bi_true is not None:
            B_bi_true = B_bi_true.astype(int)

        np.savetxt(join(output_dir,'W_true.csv'), W_true, delimiter=',')
        run.log_artifact(join(output_dir,'W_true.csv'))
        for i, A_i_true in enumerate(W_lags_true):
            np.savetxt(join(output_dir,f'A{i}_true.csv'), A_i_true, delimiter=',')
            run.log_artifact(join(output_dir,f'A{i}_true.csv'))
        if W_bi_true is not None:
            np.savetxt(join(output_dir,'W_bi_true.csv'), W_bi_true, delimiter=',')
            run.log_artifact(join(output_dir,'W_bi_true.csv'))

        np.savetxt(join(output_dir,'X.csv'), X, delimiter=',')
        run.log_artifact(join(output_dir,'X.csv'))

        for i, Y_i in enumerate(Y):
            np.savetxt(join(output_dir,f'Y{i+1}.csv'), Y_i, delimiter=',')
            run.log_artifact(join(output_dir,f'Y{i+1}.csv'))

        run.log_metric('n', n)
        run.log_metric('d', d)
        run.log_metric('p', p)
        #run.log_param('algo', cfg.solver.name)

        #run.log_param('seed', seed)

        logger.info(f'DATA GENERATED')

        # if lambda1 is not None:
        #     run.log_param('lambda1', lambda1)
        # if lambda2 is not None:
        #     run.log_param('lambda2', lambda2)
        # run.log_param('loss_type', loss_type)

        logger.info(f'STARTING SOLVER')
        start_time = time.time()
        W_est = None
        W_bi_est = None
        B_est, B_bi_est = None, None
        if cfg.solver.name == 'milp':
            W_est, A_est, gap, lazy_count, stats = solve_milp.solve(X, cfg.solver, cfg.solver.nonzero_threshold, Y=Y,
                                                                    B_ref=B_true,
                                                                    tabu_edges=tabu_edges if cfg.solver.tabu_edges else None)
            B_est = (W_est != 0).astype(int)
            if W_est is None:
                run.log_text(f'Error: Gurobi has not found a solution', 'error.txt')
                run.log_metric('infeasible', True)
                return
            if stats:
                table_data = {}
                table_data['Time'] = [s[0] for s in stats]
                table_data['Def_thresh_SHD'] = [s[1] for s in stats]
                table_data['Best_SHD'] = [s[2] for s in stats]
                table_data['Best_threshold'] = [s[3] for s in stats]
                table_data['Objective_val'] = [s[4] for s in stats]
                table_data['Dag_t'] = [s[5] for s in stats]
                run.log_table(table_data, 'solving_progress.json')
            run.log_metric('mipgap', gap)
            run.log_metric('lazy_added', lazy_count)
        elif cfg.solver.name == "exmag":
            callback_mode = cfg.solver.callback_mode
            robust = cfg.solver.robust
            weights_bound = cfg.solver.weights_bound
            reg_type = cfg.solver.reg_type
            run.log_param('reg_type', reg_type)
            a_reg_type = cfg.solver.a_reg_type
            run.log_param('a_reg_type', a_reg_type)
            target_mip_gap = cfg.solver.target_mip_gap
            run.log_param('target_mip_gap', target_mip_gap)
            run.log_param('robust', robust)
            run.log_param('callback_mode', callback_mode)
            run.log_param('weights_bound', weights_bound)

            exmag_version = cfg.solver.get('exmag_version', 'version0')

            if exmag_version == 'version0':
                W_est, W_bi_est, gap, lazy_count, stats = solve_exmag.solve(X, cfg.solver.lambda1, cfg.solver.loss_type, reg_type,
                                                                       cfg.solver.nonzero_threshold, tabu_edges=tabu_edges,
                                                                       B_ref=B_true, mode=callback_mode,
                                                                       time_limit=cfg.solver.time_limit,
                                                                       robust=robust, weights_bound=weights_bound)
            elif exmag_version == 'version2':
                from dagsolvers import solve_exmag_2
                W_est, W_bi_est, gap, lazy_count, stats = solve_exmag_2.solve(X, cfg.solver.lambda1, cfg.solver.loss_type,
                                                                            reg_type,
                                                                            cfg.solver.nonzero_threshold,
                                                                            tabu_edges=tabu_edges,
                                                                            B_ref=B_true, mode=callback_mode,
                                                                            time_limit=cfg.solver.time_limit,
                                                                            robust=robust, weights_bound=weights_bound)

            B_est = (W_est != 0).astype(int)
            B_bi_est = (W_bi_est != 0).astype(int)
            A_est = []
            # (X, lambda1, loss_type, reg_type, w_threshold, tabu_edges={}, B_ref=None, mode='shortest_cycle',
            #           time_limit=300, robust=False, weights_bound=100.0, constraints_mode='weights')
            if W_est is None:
                run.log_text(f'Error: Gurobi has not found a solution', 'error.txt')
                run.log_metric('infeasible', True)
                return
            if stats:
                table_data = {}
                table_data['Time'] = [s[0] for s in stats]
                table_data['Def_thresh_SHD'] = [s[1] for s in stats]
                table_data['Best_SHD'] = [s[2] for s in stats]
                table_data['Best_threshold'] = [s[3] for s in stats]
                table_data['Objective_val'] = [s[4] for s in stats]
                table_data['Dag_t'] = [s[5] for s in stats]
                run.log_table(table_data, 'solving_progress.json')
            run.log_metric('mipgap', gap)
            run.log_metric('lazy_added', lazy_count)
        elif cfg.solver.name == 'gobnilp':
            palim = cfg.solver.get('palim')
            run.log_param('palim', palim)
            print('palim')
            print(palim)
            W_est = solve_gobnilp.solve(X, cfg.solver)
            A_est = []
        elif cfg.solver.name == 'ip4ancadmg':
            from dagsolvers.solvers.IP4AncADMG.solver_ip4ancadmg import solve_ip4ancadmg
            B_est, B_bi_est = solve_ip4ancadmg(X, cfg.solver)
            W_est = B_est
            W_bi_est = B_bi_est
            A_est = []
        elif cfg.solver.name == 'ipbm':
            B_est, B_bi_est = solve_ipbm(X, cfg.solver)
            W_est = B_est
            W_bi_est = B_bi_est
            A_est = []
        elif cfg.solver.name == 'micodag':
            # pip install micodag
            import dagsolvers.solvers.micpnid as mic
            moral = [(i+1,j+1) for i in range(d) for j in range(d) if i!=j]
            RGAP, W_est, _, obj, _ = mic.optimize(X, moral, cfg.solver.lambda1, timelimit = cfg.solver.time_limit / d)
            run.log_metric('objective_value', obj)
            run.log_metric('mipgap', RGAP)
            A_est = []
        elif cfg.solver.name == 'notears':
            W_est = linear.notears_linear(X, cfg.solver.lambda1, cfg.solver.loss_type, w_threshold=cfg.solver.nonzero_threshold) # Does not work well with 0.
            B_est = (W_est != 0).astype(int)
            A_est = []
        elif cfg.solver.name == 'sortnregress':
            W_est = sortnregress(X)
            A_est = []
        elif cfg.solver.name == 'dagma':
            from solve_dagma import solve_dagma
            W_est = solve_dagma(X, cfg.solver)
            A_est = []
        elif cfg.solver.name == 'fci':
            B_est, B_bi_est = solve_fci(X)
            W_est = None
            W_bi_est = None
            A_est = []
        elif cfg.solver.name == 'boss':
            from solve_boss import solve_boss
            W_est = solve_boss(X, output_dir, cfg.solver.cmd)
            A_est = []
        elif cfg.solver.name == 'lingam':
            from solve_lingam import solve_lingam
            W_est, A_est = solve_lingam(X, Y, p)
        elif cfg.solver.name == 'nts_notears':
            from solve_nts_notears import solve_nts_notears
            W_est, A_est = solve_nts_notears(X, Y, p, cfg.solver)
        elif cfg.solver.name == 'dynotears':
            X_lag = np.concatenate(Y, axis=1)
            _, W_est, A_est_concated = from_numpy_dynamic(X,X_lag, w_threshold=cfg.solver.nonzero_threshold, lambda_w=cfg.solver.lambda1, lambda_a=cfg.solver.lambda2)
            A_est = []
            for lag in range(p): # range(1, p + 1):
                #idxs = [f'_lag{lag}' in c for c in inter_nodes]
                idxs = list(range(d*lag, d*(lag+1)))
                A_est_lag = A_est_concated[idxs,:]
                A_est.append(A_est_lag)
            #print(A_est)
        else:
            assert False
        solving_duration = time.time() - start_time
        logger.info(f'SOLVER FINISHED')
        if W_est is not None and not utils.is_dag(W_est):
            run.log_text('Error: Graph found is not DAG', 'error.txt')
            run.log_metric('infeasible', True)
            return
        run.log_metric('runtime', solving_duration)
        #assert utils.is_dag(W_est)
        if W_est is not None:
            np.savetxt(join(output_dir,'W_est.csv'), W_est, delimiter=',')
            run.log_artifact(join(output_dir,'W_est.csv'))

        with open(join(output_dir,'intra_nodes.txt'), "w") as f:
            s = "[" + ",".join(intra_nodes) + "]"
            f.write(s)
        run.log_artifact(join(output_dir,'intra_nodes.txt'))
        with open(join(output_dir,'inter_nodes.txt'), "w") as f:
            s = "[" + ",".join(inter_nodes) + "]"
            f.write(s)
        run.log_artifact(join(output_dir,'inter_nodes.txt'))

        if W_bi_est is not None:
            np.savetxt(join(output_dir,'W_bi_est.csv'), W_bi_est, delimiter=',')
            run.log_artifact(join(output_dir,'W_bi_est.csv'))
        for i, A_est_i in enumerate(A_est):
            np.savetxt(join(output_dir,f'A_est_{i}_no_t.csv'), W_est, delimiter=',')
            run.log_artifact(join(output_dir,f'A_est_{i}_no_t.csv'))


        # B and B_bi
        # B[i,j] = 1 :::: i --> j :::: direct cause
        # B[i,j] = 2 ::::  i o-> j :::: direct cause or common confounder
        # B_bi[i,j] = 1 :::: i <-> j :::: common confounder
        # B_bi[i,j] = 2 :::: i --- j :::: selection bias - common effect
        # B_bi[i,j] = 3 :::: i o-o j :::: direct cause either direction or common confounder or common effect

        if W_est is None:
            assert B_est is not None
            calculate_metrics_pag(B_true, B_bi_true, B_est, B_bi_est, run, cfg)

        else:
            if B_est is not None:
                assert not (B_est == 2).any() or (B_bi_est == 3).any() or (B_bi_est == 2).any(), 'PAG with weights - Not correctly implemented yet'
            best_W, best_Wbi, best_A = calculate_metrics(X, Y, W_true, B_true, W_lags_true, B_lags_true, W_est, A_est, W_bi_true, B_bi_true, W_bi_est, run, cfg)

            if cfg.plot:
                make_plots(run, W_true, W_est, best_W, W_bi_true, W_bi_est, best_Wbi, intra_nodes, W_lags_true, A_est, best_A, inter_nodes, output_dir, cfg.plot_dpi)

if __name__ == "__main__":
    start_experiment()



