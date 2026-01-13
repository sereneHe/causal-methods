from os.path import join

import pandas as pd

from data_load_utils import load_data, prune_and_clean_data, get_problem_columns_names, get_solver_columns_names, \
    filter_best
from results.data_load_utils import aggregate_data, convert_to_numeric, save_dataframe, load_dataframe
from results.plot_utils import make_grid_problem_samples_plot

def filter_data(df):
    df = df.loc[df['params.solver.name'] != 'milp']
    df = df.loc[df['params.problem.number_of_variables'] != '25']
    df = df.loc[(df['params.problem.edge_ratio'] == '2') | (df['params.problem.name'] != 'ermag')]
    df = df.loc[(df['params.solver.name'] != 'exmag') | (df['params.solver.time_limit'] == '900')]
    #df = df.loc[df['params.solver.name'] == 'fci']
    return df





def generate_metrics_plots(df, output_dir, out_prefix):

    columns_m = ['metrics.best_f1score', 'metrics.best_shd', 'metrics.runtime', 'metrics.t0.5_shd','metrics.t0.15_shd','metrics.t0.5_f1score','metrics.t0.15_f1score',]
    problem_columns = get_problem_columns_names(df)
    solver_columns = get_solver_columns_names(df)
    columns_p = problem_columns + solver_columns
    columns_to_plot_default = ['params.problem.name', 'params.problem.pdir', 'params.problem.pbidir', 'params.problem.edge_ratio', 'params.problem.max_in_arrows']
    columns_to_plot_in_data = set(problem_columns)
    columns_p_plot = [c for c in columns_to_plot_default if c in columns_to_plot_in_data]
    df = prune_and_clean_data(df)
    df = aggregate_data(df, columns_p, columns_m)
    #df = filter_best(df, problem_columns, solver_columns, 'metrics.best_shd_mean')
    df = convert_to_numeric(df, ['params.problem.number_of_variables', 'params.problem.number_of_samples'])

    def problem_names(x):
        type = x['params.problem.name']
        if type == 'ermag':
            s = f'ER-{x["params.problem.edge_ratio"]}'
        else:
            s = f'BF-{x["params.problem.pdir"]}-{x["params.problem.pbidir"]}'
            if 'params.problem.max_in_arrows' in x:
                max_in_arrows = f'{x['params.problem.max_in_arrows']}'
                if max_in_arrows != 'nan' and max_in_arrows != 'None':
                    s = f'{x['params.problem.max_in_arrows']}'+s
        return s


    for metric_to_plot, label  in [('metrics.best_shd', 'Structural Hamming distance - SHD'), ('metrics.t0.5_shd', 'Structural Hamming distance - SHD'), ('metrics.t0.15_shd', 'Structural Hamming distance - SHD'),('metrics.best_f1score', 'F1 score'),('metrics.t0.15_f1score', 'F1 score'), ('metrics.t0.5_f1score', 'F1 score'), ('metrics.runtime', 'Run time')]:
        plot_min = 'f1score' in metric_to_plot
        plot_max = 'shd' in metric_to_plot
        make_grid_problem_samples_plot(df, metric_to_plot, label, output_dir, out_prefix, columns_p_plot, plot_max, plot_min, problem_names)


def generate_exmag_graphs(reload_data=False):
    output_dir = '/Users/pavel/0_code/0_causal/data_exdag_exdbn/EXMAG_ICLR_2025_graphs'
    if reload_data:
        data = [(['EXMAG3'],'mlflow_planserver', '')]#, (['exmag_iclr2025'], 'parquet', '')] #
        df, _ = load_data(data,csv_file_path='/Users/pavel/0_code/0_causal/data_exdag_exdbn/EXMAG_ICML_2025_graphs')

        data = [(['EXMAG'],'mlflow_planserver', '')]#, (['exmag_iclr2025'], 'parquet', '')] #
        df_ipbm, _ = load_data(data,csv_file_path='/Users/pavel/0_code/0_causal/data_exdag_exdbn/EXMAG_ICML_2025_graphs')
        df_ipbm = df_ipbm.loc[df_ipbm['params.solver.name'] == 'ipbm']

        df = pd.concat([df, df_ipbm])

        save_dataframe(df, join(output_dir, 'exmag_iclr2025'))
    else:
        df = load_dataframe(join(output_dir, 'exmag_iclr2025'))
    #df = df.loc[df['params.solver.name'] == 'fci']
    #df = filter_data(df)
    df1 = df.loc[df['params.problem.max_in_arrows'] != '3']
    generate_metrics_plots(df1, output_dir, f'iclr2025')
    df2 = df.loc[df['params.problem.max_in_arrows'] == '3']
    generate_metrics_plots(df2, output_dir, f'iclr2025_2')
    df3 = df.loc[(df['params.problem.edge_ratio'] == '2') | (df['params.problem.name'] != 'ermag')]
    df3 = df3.loc[df3['params.problem.max_in_arrows'] != '3']
    generate_metrics_plots(df3, output_dir, f'iclr2025_3')

if __name__ == "__main__":

    generate_exmag_graphs()



