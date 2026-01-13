import os
from typing import Optional, Tuple, List

import pandas as pd
import seaborn as sns
from os.path import join

from matplotlib import pyplot as plt

BASE_DIR = "/home/rysavpe1/codiet/exmag/exmagres/"

output_dir = '/Users/pavel/0_code/0_causal/data_exdag_exdbn/ICLR2025_graphs'

def create_convergence_graphs(df, outputfile:str):
    columns_p = ['metrics.p', 'params.solver.lambda1', 'params.solver.lambda2', 'params.solver.name', 'metrics.d', 'metrics.n', 'params.a_reg_type', 'params.problem.seed', 'params.solver.time_limit']
    print(f'Number of finished: {len(df.index)}')
    columns_groupby = list(columns_p)
    columns_groupby.remove('params.problem.seed')
    columns_m = ['metrics.best_f1score', 'metrics.best_shd', 'metrics.best_shd_var', 'metrics.runtime', 'metrics.best_gscore', 'metrics.best_norm_distance', ]
    df = df[columns_p + columns_m]
    print(df.head().to_string())
    data_to_plot = df
    data_to_plot.rename(columns={'params.d': 'Variables'}, inplace=True)

    pic = sns.relplot(x='metrics.runtime', y='metrics.best_shd',
                      #hue='Variables',
                      col='metrics.n', data=data_to_plot, kind='line', col_wrap=2, height=3, palette="tab10")
    pic.set_axis_labels('Time limit', 'SHD')
    pic.set_titles('Number of samples: {col_name}')
    #leg = pic._legend
    #leg.set_bbox_to_anchor([0.5, 0.7])
    #pic.set(xticks=[5, 10, 15,20, 25])
    #pic._legend.texts[0].set_text('Number of variables')

    figure = pic.figure
    #figure.suptitle("SF3-1")
    figure.savefig(f'{BASE_DIR}ICML-convergence-graph.png', bbox_inches='tight')


def create_plot(df, outputfile:str, dynamic_networks, only_best: bool = True):

    # df = pd.read_csv(f'{outputfile}.csv')
    # df_failed = df.loc[df['status'] == 'FAILED']
    # print(f'Number of failed: {len(df_failed.index)}')
    # df['metrics.mipgap'] = pd.to_numeric(df['metrics.mipgap'].fillna(1000))
    # df = df.fillna('')
    # df = df.loc[df['status'] == 'FINISHED']
    # df = df.loc[(df['params.algo'] != 'milp') | (df['metrics.mipgap'] < 1)]

    # find best params
    columns_p = ['metrics.p', 'params.solver.lambda1', 'params.solver.lambda2', 'params.solver.name', 'metrics.d', 'metrics.n', 'params.a_reg_type', 'params.problem.seed', 'params.solver.time_limit', 'params.solver.loss_type']


    # if p is not None:
    #     df = df.loc[df['params.p'] == p]
    #df = df.loc[(df['params.n'] == 500) & (df['params.d'] <= 40)]
    #df = df.loc[(df['params.n'] > 30)]


    print(f'Number of finished: {len(df.index)}')
    columns_groupby = list(columns_p)
    columns_groupby.remove('params.problem.seed')
    columns_m = ['metrics.best_f1score', 'metrics.best_shd', 'metrics.best_shd_var', 'metrics.runtime', 'metrics.best_gscore', 'metrics.best_norm_distance', ]
    df = df[columns_p + columns_m]
    print(df.head().to_string())






    metrics_to_plot = ['metrics.best_shd', 'metrics.best_norm_distance', 'metrics.best_f1score', 'metrics.best_gscore'] # 'metrics.best_shd_var',
    metrics_agg = ['max', 'max', 'min', 'min'] # 'max',
    metrics_label = ['SHD', 'L2 DIST', 'F1 score', 'G score'] # 'SHD',
    agg_select = 'max' # min
    df_grouped = df.groupby(columns_groupby)
    df_mean = df_grouped[metrics_to_plot].agg(['mean','min', 'max']).stack().reset_index().rename(columns={'level_9': 'Aggregation'}) #, 0: 'vals'})
    print(df_mean.to_string())

    columns_p_a = list(columns_groupby)
    columns_p_a.remove('metrics.d')
    df_mean_grouped = df_mean.loc[df_mean['Aggregation'] == 'max'].drop("Aggregation", axis='columns').groupby(columns_p_a).mean().stack().reset_index().rename(columns={ 0: 'vals'})
    df_mean_grouped = df_mean_grouped.loc[df_mean_grouped['level_8'] == 'metrics.best_shd']
    print(df_mean_grouped.to_string())

    if only_best:
        idx = df_mean_grouped.groupby(['metrics.n', 'params.solver.name'])['vals'].idxmin()
        df_mean_grouped = df_mean_grouped.loc[idx]
        print(df_mean_grouped.to_string())

        df_merged = pd.merge(left=df_mean_grouped, right=df_mean, how='left', left_on=columns_p_a, right_on=columns_p_a)
        print(df_merged.to_string())
        df_mean = df_merged

    for y_col, agg, label in zip(metrics_to_plot, metrics_agg, metrics_label):

        col_name = label #
        file_suffix = y_col.split('.',1)[1]
    # shd
        data_to_plot = df_mean.loc[(df_mean['Aggregation'] == agg) | (df_mean['Aggregation'] == 'mean')]

        max_number_vars = ((round(max(data_to_plot['metrics.d'])) // 5) + 1) * 5
        min_number_vars = (round(min(data_to_plot['metrics.d'])) // 5) * 5


        pic = sns.relplot(x='metrics.d', y=y_col, style='Aggregation',
                          hue=data_to_plot[['params.solver.lambda1','params.solver.lambda2','params.a_reg_type','params.solver.name', 'params.solver.time_limit', 'params.solver.loss_type']].apply(tuple, axis=1),
                          col='metrics.n', data=data_to_plot, kind='line', col_wrap=2, height=2)

        # for ax in pic.axes.flatten():
        #     for i, l in enumerate(ax.get_lines()):
        #         if i == 1 or i == 0:
        #             l.set_color('black')

        pic.set_axis_labels('Number of variables', col_name)
        pic.set_titles('Samples: {col_name}')
        pic.set(xticks=list(range(min_number_vars, max_number_vars + 1, 10)))
        pic._legend.texts[0].set_text('Algorithm')
        for i in range(len(pic._legend.texts)):
            orig_text = pic._legend.texts[i].get_text()
            if dynamic_networks: # dynodag
                if 'dynotears' in orig_text:
                    pic._legend.texts[i].set_text('dynotears')
                if 'lingam' in orig_text:
                    pic._legend.texts[i].set_text('lingam')
                elif 'milp' in orig_text:
                    orig_text = orig_text[1:-1]
                    orig_text_split = orig_text.split(',')
                    new_text = f'Ex({orig_text_split[0][1:-1]}, {orig_text_split[1][2:-1]}, {orig_text_split[2][2:-1]})' #, {orig_text_split[5][2:-1]})'
                    # new_text = f'ExDBN λ={orig_text_split[0]} η={orig_text_split[1]} reg={orig_text_split[2][2:-1]}'
                    pic._legend.texts[i].set_text(new_text)
            else:
                if 'notears' in orig_text:
                    pic._legend.texts[i].set_text('notears')
                elif 'boss' in orig_text:
                    pic._legend.texts[i].set_text('boss')
                elif 'dagma' in orig_text:
                    pic._legend.texts[i].set_text('dagma')
                elif 'milp' in orig_text:
                    orig_text = orig_text[1:-1]
                    orig_text_split = orig_text.split(',')
                    new_text = f'ExDAG λ={orig_text_split[0]}'
                    pic._legend.texts[i].set_text(new_text)
        leg = pic._legend
        leg.set_bbox_to_anchor([0.52, 0.18]) # for 5 subgraphs 56 - moc napravo 46 moc nalevo
        #leg.set_bbox_to_anchor([0.25, 1.2])

        figure = pic.figure
        #figure.suptitle("SF3-1")
        figure.savefig(join(output_dir,f'{BASE_DIR}_{file_suffix}.png'), bbox_inches='tight')
        plt.close(figure)


    # # norm distance
    # pic = sns.relplot(x='params.d', y='metrics.best_norm_distance', style='Aggregation', hue=df_mean[['params.lambda1','params.lambda2','params.a_reg_type','params.algo', 'params.time_limit']].apply(tuple, axis=1), col='params.n', data=df_mean, kind='line', col_wrap=2)
    #
    # pic.set_axis_labels('# Variables', 'NORM_DIST')
    # pic.set_titles('Number of samples: {col_name}')
    # figure = pic.figure
    # figure.savefig(outputfile+'_norm.png', bbox_inches='tight')

    # df_max = df_grouped.max()
    #
    # df_mean = df_mean.unstack(level=['params.p', 'parproblem.w_min_inter=0.3 problem.w_max_inter=0.5ams.lambda1', 'params.algo']) #,'params.n'])
    # print(df_mean.head())

    # pic = sns.relplot(data=df_mean)
    # figure = pic.figure
    # figure.savefig(outputfile, bbox_inches='tight')

    # ax = df_mean.groupby('params.n').plot() #(x='params.d',y='metrics.best_shd')
    # ax0 = ax.iloc[0]
    # fig = ax.iloc[1].get_figure()
    # fig.savefig(outputfile)

    # ax = df_mean.plot() #(x='params.d',y='metrics.best_shd')
    # fig = ax.get_figure()
    # fig.savefig(outputfile)


def load_data(data: List[Tuple[List[str],str]], csv_file_path='./'):
    dfs = []
    dfs_failed = []
    for experiment_names, data_source in data:
        if 'mlflow' in data_source:
            import mlflow

            if data_source == 'mlflow':
                df = mlflow.search_runs(experiment_names=experiment_names) #, filter_string="status = 'FINISHED' and params.solver = 'miqp' and params.time_limit = '43200' and params.noise = 'gauss_fault'")
            else:
                env_prefix = data_source.split('_')[1]
                import mlflow.environment_variables as mlenv
                old_uri = mlflow.get_tracking_uri()

                old_user = mlenv.MLFLOW_TRACKING_USERNAME.get()
                mlenv.MLFLOW_TRACKING_USERNAME.set(os.environ[f'{env_prefix}_MLFLOW_TRACKING_USERNAME']) # TODO: move this to a config file
                old_password = mlenv.MLFLOW_TRACKING_PASSWORD.get()
                mlenv.MLFLOW_TRACKING_PASSWORD.set(os.environ[f'{env_prefix}_MLFLOW_TRACKING_PASSWORD'])
                old_tls = mlenv.MLFLOW_TRACKING_INSECURE_TLS.get()
                mlenv.MLFLOW_TRACKING_INSECURE_TLS.set(True)
                mlflow.set_tracking_uri(os.environ[f'{env_prefix}_MLFLOW_TRACKING_URI'])

                df = mlflow.search_runs(experiment_names=experiment_names,  max_results=4000, order_by=["tag.end_time DESC"], filter_string="status = 'FINISHED' and params.variant='sf' and params.algo='milp' and params.n = '250' and params.d='25' and params.p='1' params.time_limit != '7200'") #


                mlflow.set_tracking_uri(old_uri)
                if old_user is None:
                    mlenv.MLFLOW_TRACKING_USERNAME.unset()
                else:
                    mlenv.MLFLOW_TRACKING_USERNAME.set(old_user)
                if old_password is None:
                    mlenv.MLFLOW_TRACKING_PASSWORD.unset()
                else:
                    mlenv.MLFLOW_TRACKING_PASSWORD.set(old_password)
                if old_tls is None:
                    mlenv.MLFLOW_TRACKING_INSECURE_TLS.unset()
                else:
                    mlenv.MLFLOW_TRACKING_INSECURE_TLS.set(old_tls)

            df_failed = df.loc[df['status'] == 'FAILED']
            df = df.loc[df['status'] == 'FINISHED']
        elif data_source == 'csv':
            df_loaded = [pd.read_csv(join(csv_file_path,f'{experiment_name}.csv')) for experiment_name in experiment_names]
            df_loaded = [rename_neptune_columns(df) for df in df_loaded]
            df = pd.concat(df_loaded)
            df_failed = df.loc[df['status'] == 'FAILED']
            df = df.loc[df['status'] == 'FINISHED']
        else:
            assert False
        dfs.append(df)
        dfs_failed.append(df_failed)
    df = pd.concat(dfs)
    df_failed = pd.concat(dfs_failed)
    print(f'Number of failed: {len(df_failed.index)}')
    print(f'Number of finished: {len(df.index)}')
    return df, df_failed


def prune_and_clean_data(df):
    if 'metrics.infeasible' in df.columns:
        df = df.loc[df['metrics.infeasible'] != 1]
    for column in ['params.edge_ratio', 'params.p', 'params.intra_edge_ratio','params.inter_edge_ratio', 'params.d', 'params.n']:
        if column in df.columns:
            print(column)
            df[column] = pd.to_numeric(df[column]).round(0).astype(int)

    df['metrics.mipgap'] = pd.to_numeric(df['metrics.mipgap'].fillna(1000))
    df = df.fillna('')

    if 'params.noise_scale_variance' in df.columns:
        df['params.noise_scale_variance'] = df['params.noise_scale_variance'].replace('None', '')
    if 'params.normalize' in df.columns:
        df['params.normalize'] = df['params.normalize'].replace('False', '')

    df = df.loc[(df['params.solver.name'] != 'exmag') | (df['metrics.mipgap'] < 1)]
    df = df.loc[(df['params.problem.name'] != '')]
    df = df.loc[(df['metrics.best_shd'] != '')]
    print(f'Number of finished after cleaning: {len(df.index)}')
    return df

def generate_graphs(df, dynamic_networks, max_d: Optional[int] = None, name_prefix='EXP'):

    if max_d is not None:
        df = df.loc[(df['params.d'] <= max_d)]

    columns_groupby = ['params.problem', 'params.edge_ratio', 'metrics.p', 'params.variant', 'params.intra_edge_ratio',
                       'params.inter_edge_ratio', 'params.normalize', 'params.noise_scale', 'params.noise_scale_variance']
    df_columns = set(df.columns.values)
    columns_groupby = [c for c in columns_groupby if c in df_columns]
    gb = df.groupby(columns_groupby)
    dfs = [(key, gb.get_group(key)) for key in gb.groups]

    for key, _ in dfs:
        print(key)

    for key, group_df in dfs:
        name = "-".join(f'{k}'.replace('.','_') for k in key if k != '')
        if max_d is not None:
            name += f'_maxv{max_d}'
        print(name)

        create_plot(group_df, f'{name_prefix}-{name}', dynamic_networks)
        #create_convergence_graphs(group_df, f'{experiment_names[0]}-{name}')



def rename_neptune_columns(df):
    columns_to_rename = {c: c.replace('/','.') for c in df.columns.values if ('params' in c) or ('metrics' in c)}
    df.rename(columns=columns_to_rename, inplace=True)
    #df.rename(columns={'st'}, inplace=True)
    return df


def generate_iclr_graphs():
    data = [(['EXDBN'],'mlflow_planserver'),(['EXDBN'],'mlflow'), (['DYNAMIC1800', 'DYNAMIC2300', 'DYNAMIC2400'],'csv')]
    df, _ = load_data(data, csv_file_path='/Users/pavel/0_code/0_causal/data_exdag_exdbn')
    df = prune_and_clean_data(df)
    generate_graphs(df, dynamic_networks=True, name_prefix='ICLR2025')


def generate_ICML_convergence_graphs():
    data = [(['STATIC30'], 'csv')]
    df, _ = load_data(data, csv_file_path=BASE_DIR)
    df = prune_and_clean_data(df)
    create_convergence_graphs(df, join(output_dir, 'time_convergence.png'))
    generate_graphs(df, dynamic_networks=True, name_prefix='ICML2025')



if __name__ == "__main__":
    generate_ICML_convergence_graphs()
    #generate_iclr_graphs()

    # df = pd.read_csv(f'STATIC3.csv')
    # df = df.loc[((df['params.problem'] == 'sf') & ((df['params.edge_ratio'] == 3) | (df['params.edge_ratio'] == 4))) | ((df['params.problem'] == 'er') & ((df['params.edge_ratio'] == 2) ))]
    # df.to_csv(f'STATIC3COR.csv')

    #generate_graphs(['STATIC2','STATIC3COR', 'STATIC4', 'STATIC6'])
    #generate_graphs(['STATICBIG12'],max_d=80) # (['STATICCONV1'])
    #generate_graphs(['STATICBIG1'])
