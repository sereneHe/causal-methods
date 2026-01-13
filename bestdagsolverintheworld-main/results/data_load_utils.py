import os
from os.path import join
from typing import List, Tuple

import pandas as pd

from results.extract_experiments_neptune_data import fetch_neptune_data, rename_neptune_columns
from results.mlflow_utils import convert_to_df


def load_experiments_from_csv(filename):
    prefix = 'params'
    df_preview = pd.read_csv(filename, nrows=0)
    columns = df_preview.columns
    string_columns = [col for col in columns if col.startswith(prefix)]
    dtype_dict = {col: str for col in string_columns}
    df = pd.read_csv(filename, dtype=dtype_dict)
    return df


def save_dataframe(df, filename):
    df.to_parquet(f'{filename}.parquet', engine="pyarrow", compression="gzip")

def load_dataframe(filename):
    return pd.read_parquet(f'{filename}.parquet')

def get_experiments_ids(experiment_names, client):
    import mlflow
    experiments_ids = []
    for experiment_name in experiment_names:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is not None:
            experiments_ids.append(experiment.experiment_id)
        else:
            raise ValueError(f"Experiment with name '{experiment_name}' does not exist.")
    return experiments_ids


def paginated_mlflow_search(experiment_names, filter_string='', batch_size=1000):
    import mlflow
    client = mlflow.tracking.MlflowClient()
    experiment_ids = get_experiments_ids(experiment_names, client)
    runs = []
    page_token = None
    i=0
    while True:
        result = client.search_runs(
            experiment_ids=experiment_ids,
            filter_string=filter_string,
            max_results=batch_size,
            page_token=page_token
        )
        i += 1
        print(f'Fetched {i*batch_size} experiments runs.')
        runs.extend(result.to_list())
        page_token = result.token
        if not page_token:
            break
    df = convert_to_df(runs)
    return df

def load_data(data: List[Tuple[List[str],str, str]], csv_file_path='./'):
    dfs = []
    dfs_failed = []
    for experiment_names, data_source, filter_query in data:
        if data_source == 'neptune':
            df = fetch_neptune_data(experiment_names)
            df = rename_neptune_columns(df)
            df_failed = df.loc[df['sys/failed'] == True]
            df = df.loc[df['sys/state'] == 'Inactive']
            df = df.loc[df['sys/failed'] == False]
        elif 'mlflow' in data_source:
            import mlflow

            if data_source == 'mlflow':
                df = mlflow.search_runs(experiment_names=experiment_names, filter_string=filter_query) #, filter_string="status = 'FINISHED' and params.solver = 'miqp' and params.time_limit = '43200' and params.noise = 'gauss_fault'")
                if not df.empty:
                    df = prune_and_clean_data(df)
                    df = rename_columns_in_old_data(df)
            else:
                env_prefix = data_source.split('_')[1]
                import mlflow.environment_variables as mlenv
                old_uri = mlflow.get_tracking_uri()

                old_user = mlenv.MLFLOW_TRACKING_USERNAME.get()
                mlenv.MLFLOW_TRACKING_USERNAME.set(os.environ[f'{env_prefix}_MLFLOW_TRACKING_USERNAME']) # TODO: move this to a config file
                old_password = mlenv.MLFLOW_TRACKING_PASSWORD.get()
                mlenv.MLFLOW_TRACKING_PASSWORD.set(os.environ[f'{env_prefix}_MLFLOW_TRACKING_PASSWORD'])
                #old_tls = mlenv.MLFLOW_TRACKING_INSECURE_TLS.get()
                #mlenv.MLFLOW_TRACKING_INSECURE_TLS.set(True)

                old_cert_path = mlenv.MLFLOW_TRACKING_SERVER_CERT_PATH.get()
                mlenv.MLFLOW_TRACKING_SERVER_CERT_PATH.set(os.environ[f'{env_prefix}_MLFLOW_TRACKING_SERVER_CERT_PATH'])

                mlflow.set_tracking_uri(os.environ[f'{env_prefix}_MLFLOW_TRACKING_URI'])

                df = paginated_mlflow_search(experiment_names, filter_string=filter_query)
                #df = mlflow.search_runs(experiment_names=experiment_names) #,  max_results=4000, order_by=["tag.end_time DESC"], filter_string="status = 'FINISHED' and params.variant='sf' and params.algo='milp' and params.n = '250' and params.d='25' and params.p='1' params.time_limit != '7200'") #


                mlflow.set_tracking_uri(old_uri)
                if old_user is None:
                    mlenv.MLFLOW_TRACKING_USERNAME.unset()
                else:
                    mlenv.MLFLOW_TRACKING_USERNAME.set(old_user)
                if old_password is None:
                    mlenv.MLFLOW_TRACKING_PASSWORD.unset()
                else:
                    mlenv.MLFLOW_TRACKING_PASSWORD.set(old_password)
                if old_cert_path is None:
                    mlenv.MLFLOW_TRACKING_SERVER_CERT_PATH.unset()
                else:
                    mlenv.MLFLOW_TRACKING_SERVER_CERT_PATH.set(old_cert_path)
                df = rename_columns_in_old_data(df)
                df = prune_and_clean_data(df)

            df_failed = df.loc[df['status'] == 'FAILED']
            df = df.loc[df['status'] == 'FINISHED']

        elif data_source == 'csv':
            df_loaded = [load_experiments_from_csv(join(csv_file_path,f'{experiment_name}.csv')) for experiment_name in experiment_names]
            df_loaded = [rename_neptune_columns(df) for df in df_loaded]
            df_loaded = [prune_and_clean_data(df) for df in df_loaded]
            df_loaded = [rename_columns_in_old_data(df) for df in df_loaded]
            df = pd.concat(df_loaded)
            df_failed = df.loc[df['status'] == 'FAILED']
            df = df.loc[df['status'] == 'FINISHED']
        elif data_source == 'parquet':
            df_loaded = [pd.read_parquet(join(csv_file_path,f'{experiment_name}.parquet')) for experiment_name in experiment_names]
            df = pd.concat(df_loaded)
        else:
            assert False
        dfs.append(df)
        dfs_failed.append(df_failed)
    df = pd.concat(dfs)
    df_failed = pd.concat(dfs_failed)
    df = prune_and_clean_data(df)
    print(f'Number of failed: {len(df_failed.index)}')
    print(f'Number of finished: {len(df.index)}')
    return df, df_failed


def fill_from_no_threshold(df):
    for col in df.columns:
        if 'metrics.no_threshold' in col:
            m = col[len('metrics.no_threshold_'):]
            for target_col_prefix in ['metrics.best', 'metrics.t0.15', 'metrics.t0.5']:
                target_col = target_col_prefix + '_' + m
                for col2 in df.columns:
                    if target_col == col2:
                        df[target_col] = df[target_col].fillna(df[col])





def prune_and_clean_data(df):
    print(df)
    fill_from_no_threshold(df)
    #df['metrics.best_shd'] = pd.to_numeric(df['metrics.best_shd'], errors='coerce')
    df = df.dropna(subset='metrics.best_shd')
    #metrics_columns = df.filter(like='metrics.').columns
    #df[metrics_columns] = df[metrics_columns].apply(pd.to_numeric, errors='coerce')

    if 'metrics.infeasible' in df.columns:
        df = df.loc[df['metrics.infeasible'] != 1]

    assert 'params.noise_scale_variance' not in df.columns, 'OLD DATA FORMAT - just to check if it somewhere'
    if 'params.noise_scale_variance' in df.columns:
        df['params.noise_scale_variance'] = df['params.noise_scale_variance'].fillna('None').replace('', 'None')
    if 'params.problem.noise_scale_variance' in df.columns:
        df['params.problem.noise_scale_variance'] = df['params.problem.noise_scale_variance'].fillna('None').replace('', 'None')
    assert 'params.normalize' not in df.columns, 'OLD DATA FORMAT - just to check if it somewhere'
    if 'params.normalize' in df.columns:
        df['params.normalize'] = df['params.normalize'].replace('False', '')

    assert 'params.problem' not in df.columns, 'OLD DATA FORMAT - just to check if it somewhere'
    if 'params.problem' in df.columns:
        df = df.loc[(df['params.problem'] != '')]
    df = df.loc[(df['metrics.best_shd'] != '')]

    df = df.loc[(df['metrics.best_shd'] < 200)]
    print(f'Number of finished after cleaning: {len(df.index)}')
    return df


def convert_to_numeric(df, columns, to_integer=False):
    downcast = 'integer' if to_integer else None
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors='coerce', downcast=downcast)
    return df

def aggregate_data(df, columns_p, columns_m):
    columns_groupby = list(columns_p)
    columns_groupby.remove('params.problem.seed')
    df = df[columns_p + columns_m]
    df_grouped = df.groupby(columns_groupby, dropna=False)
    df_mean = df_grouped[columns_m].agg(['mean','min', 'max', 'std']).reset_index()
    df_mean.columns = ['_'.join(c for c in col if c) for col in df_mean.columns]
    #print(df_mean.to_string())
    return df_mean


def filter_best(df, columns_problem, columns_solver_params, criterium_column):
    columns_problem = list(columns_problem)
    columns_solver_params = list(columns_solver_params)
    columns_problem.remove('params.problem.number_of_variables')
    columns_problem.remove('params.problem.seed')
    #columns_problem.append('params.solver.name')

    df_vars_mean = df[columns_problem + columns_solver_params + [criterium_column]].groupby(columns_problem + columns_solver_params, dropna=False).mean().reset_index()
    #print(df_vars_mean.to_string())

    # df_mean_grouped = df_mean_max.mean().stack(future_stack=True).reset_index().rename(columns={ 0: 'vals'})
    # df_mean_grouped = df_mean_grouped.loc[df_mean_grouped['level_10'] == 'metrics.best_shd']
    # print(df_mean_grouped.to_string())

    #columns_solver_params.remove('params.solver.name')

    idx = df_vars_mean.groupby(columns_problem + ['params.solver.name'], dropna=False)[criterium_column].idxmin() # , 'params.solver.callback_mode'
    df_vars_mean = df_vars_mean.loc[idx]
    #print(df_vars_mean.to_string())
    del df_vars_mean[criterium_column]

    df_merged = pd.merge(left=df_vars_mean, right=df, how='left', left_on=columns_problem + columns_solver_params, right_on=columns_problem + columns_solver_params)
    #print(df_merged.to_string())
    return df_merged


def get_problem_columns_names(df):
    cols = [c for c in df.columns.tolist() if c.startswith('params.problem')]
    return cols


def get_solver_columns_names(df):
    cols = [c for c in df.columns.tolist() if c.startswith('params.solver')]
    return cols


def rename_columns_if_new_not_exists(df, rename_dict):
    """
    Renames columns in a DataFrame only if the new column name does not already exist.

    Parameters:
        df (pd.DataFrame): The DataFrame whose columns you want to rename.
        rename_dict (dict): A dictionary where keys are current column names and values are the intended new column names.

    Returns:
        pd.DataFrame: A DataFrame with the renamed columns, where applicable.
    """
    for old_name, new_name in rename_dict.items():
        if old_name in df.columns:
            if new_name not in df.columns:
                df.rename(columns={old_name: new_name}, inplace=True)
            else:
                print(
                    f"Skipping renaming '{old_name}' to '{new_name}' as '{new_name}' already exists in the DataFrame.")
    return df



def rename_columns_in_old_data(df):
    df = rename_columns_if_new_not_exists(df, {
        'params.noise_scale': 'params.problem.noise_scale',
        'params.noise_scale_variance': 'params.problem.noise_scale_variance',
        'params.w_max_inter': 'params.problem.w_max_inter',
        'params.w_min_inter': 'params.problem.w_min_inter',
        'params.w_decay': 'params.problem.w_decay',
        'params.sem_type': 'params.problem.sem_type',
        'params.problem': 'params.problem.name',
        'params.normalize': 'params.problem.normalize',
        'params.noise': 'params.problem.noise',
        'params.generator': 'params.problem.generator',
        'params.p': 'params.problem.p',
        'params.d': 'params.problem.number_of_variables',
        'params.seed': 'params.problem.seed',
        'params.edge_ratio': 'params.problem.edge_ratio',
        'params.inter_edge_ratio': 'params.problem.inter_edge_ratio',
        'params.intra_edge_ratio': 'params.problem.intra_edge_ratio',
        'params.variant': 'params.problem.graph_type_intra',
        'params.n': 'params.problem.number_of_samples',
        'params.lambda1': 'params.solver.lambda1',
        'params.loss_type': 'params.solver.loss_type',
        'params.target_mip_gap': 'params.solver.target_mip_gap',
        'params.lambda2': 'params.solver.lambda2',
        'params.reg_type': 'params.solver.reg_type',
        'params.a_reg_type': 'params.solver.a_reg_type',
        'params.weights_bound': 'params.solver.weights_bound',
        'params.time_limit': 'params.solver.time_limit',
        'params.robust': 'params.solver.robust',
        'params.callback_mode': 'params.solver.callback_mode',
        'params.algo': 'params.solver.name',
    })
    return df


def print_solver_params(df, solver_columns):
    print(df[solver_columns].drop_duplicates().to_string())
