from os.path import join
import pandas as pd

from data_load_utils import load_data, prune_and_clean_data, get_problem_columns_names, get_solver_columns_names, \
    rename_columns_in_old_data, filter_best, save_dataframe, load_dataframe
from results.data_load_utils import aggregate_data, convert_to_numeric
from results.plot_utils import make_grid_problem_samples_plot


def generate_exdag_graphs():
    columns_m = ['metrics.best_f1score', 'metrics.best_shd', 'metrics.runtime', ]
    file_prefix = f'icml2025'

    output_dir = '/Users/pavel/0_code/0_causal/data_exdag_exdbn/EXDAG_ICML2025_graphs'
    data = [(['EXDAG'],'mlflow_planserver', '')]
    #df, _ = load_data(data,csv_file_path='/Users/pavel/0_code/0_causal/data_exdag_exdbn')
    #df = filter_data(df)
    #save_dataframe(df, join(output_dir, 'exdag_icml2025'))
    df = load_dataframe(join(output_dir, 'exdag_icml2025'))

    df['params.solver.name'] = df['params.solver.name'].apply(lambda x : x.replace('milp', 'exdag'))

    problem_columns = get_problem_columns_names(df)
    problem_columns = [c for c in problem_columns if c not in ['params.problem.normalize', 'params.problem.p']] #

    solver_columns = get_solver_columns_names(df)
    solver_columns = [c for c in solver_columns if c not in ['params.solver.lambda2', 'params.solver.reg_type', 'params.solver.a_reg_type', 'params.solver.nonzero_threshold', 'params.solver.constraints_mode', 'params.solver.plot', 'params.solver.tabu_edges', 'params.solver.target_mip_gap', 'params.solver.time_limit', 'params.solver.weights_bound', 'params.solver.robust', 'params.solver.callback_mode',]]

    columns_p = problem_columns + solver_columns
    columns_to_plot_default = []
    columns_to_plot_default = ['params.problem.name', 'params.problem.edge_ratio', 'params.problem.noise_scale' , 'params.problem.noise_scale_variance'] # ,  'params.problem.p','params.problem.noise_scale' , 'params.problem.noise_scale_variance'
    columns_to_plot_in_data = set(problem_columns)
    columns_p_plot = [c for c in columns_to_plot_default if c in columns_to_plot_in_data]

    df = aggregate_data(df, columns_p, columns_m)
    df = filter_best(df, problem_columns, solver_columns, 'metrics.best_shd_mean')
    df = convert_to_numeric(df, ['params.problem.number_of_variables', 'params.problem.number_of_samples'], to_integer=True)
    def method_names(x):
        if x['params.problem.noise_scale_variance'] != 'None':
            noise_str = '-var'
        else:
            noise_str = ''
        s = f'{str(x["params.problem.name"]).upper()}-{x['params.problem.edge_ratio']}' + noise_str
        return s
    for metric_to_plot, label  in [('metrics.best_shd', 'Structural Hamming distance - SHD'), ('metrics.best_f1score', 'F1 score')]:
        plot_min = metric_to_plot == 'metrics.best_f1score'
        plot_max = metric_to_plot == 'metrics.best_shd'
        y_lim = 100 if metric_to_plot == 'metrics.best_shd' else None
        make_grid_problem_samples_plot(df, metric_to_plot, label, output_dir, file_prefix, columns_p_plot, plot_max, plot_min, method_names, y_lim=y_lim)


if __name__ == "__main__":

    generate_exdag_graphs()



