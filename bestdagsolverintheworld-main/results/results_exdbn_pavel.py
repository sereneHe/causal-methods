import math
from os.path import join
import polars as pl

from experiments_utils import load_polars_dataframe, replace_dots_in_columns
import experiments_utils
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import altair as alt
#import plotly.express as px


from data_load_utils import load_data, prune_and_clean_data, get_problem_columns_names, get_solver_columns_names, \
    rename_columns_in_old_data, filter_best, print_solver_params, save_dataframe, load_dataframe
from results.data_load_utils import aggregate_data, convert_to_numeric
from results.plot_utils import make_grid_problem_samples_plot


def generate_exdbn_graphs(output_dir, graph_type='solution_metrics', reload_data=False):
    import pandas as pd
    columns_m = ['metrics.best_f1score',  'metrics.best_shd', 'metrics.runtime', 'metrics.t0.5_shd', 'metrics.t0.15_shd', 'metrics.t0.15_f1score', 'metrics.t0.5_f1score', 'metrics.lazy_added']

    data = [(['EXDBN2'],'mlflow_planserver', '')]
    if reload_data:
        df, _ = load_data(data,csv_file_path='/Users/pavel/0_code/0_causal/data_exdag_exdbn')
        df = prune_and_clean_data(df)
        save_dataframe(df, join(output_dir, 'exdbn_tmlr'))
    else:
        df = load_dataframe(join(output_dir, 'exdbn_tmlr'))
    df = df[(df['params.solver.name'] != "milp") | (df["params.solver.callback_mode"] == "all_cycles")]
    # df2 = load_dataframe(join(output_dir, 'exdbn_tmlr'))
    # df = pd.concat([df, df2])

    df['params.solver.name'] = df['params.solver.name'].apply(lambda x : x.replace('milp', 'exdbn'))

    df = convert_to_numeric(df, ['params.problem.number_of_variables', 'params.problem.number_of_samples'])

    # Calculate average run time

    df_run_times = (
        df.groupby(['params.solver.name'], as_index=False)['metrics.runtime'].mean()
        .rename(columns={'run_time': 'run_time_mean'})
    )
    print(df_run_times)

    df_run_times = (
        df.groupby(['params.solver.name', 'params.problem.number_of_variables'], as_index=False)['metrics.runtime']
      .agg(['mean', 'std']).rename(columns={'mean': 'run_time_mean', 'std': 'run_time_std'})
        #.mean() # , 'params.problem.number_of_samples'
        #.rename(columns={'run_time': 'run_time_mean'})
    )
    print(df_run_times.to_string())

    df_run_times = df_run_times.pivot(
        index="params.problem.number_of_variables",
        columns="params.solver.name",
        values=['run_time_mean','run_time_std']
    )
    print(df_run_times.to_string())

    df_latex = df_run_times.copy()
    df_latex.columns = df_latex.columns.set_names(['stat', 'solver'])
    df_latex = (
        df_latex.stack(level='solver')
        .apply(lambda x: f"{x['run_time_mean']:.1f} \\pm {x['run_time_std']:.1f}", axis=1)
        .unstack(level='solver')
    )

    # export to LaTeX
    latex_table = df_latex.to_latex(escape=False)
    print(latex_table)

    # latex = df_run_times.to_latex(float_format="%.1f", index=True)
    # print(latex)

    if graph_type == 'exdbn_metrics':
        df = df[(df['params.solver.name'] == "exdbn")]

        #print(df.to_string())
    problem_columns = get_problem_columns_names(df)
    # problem_columns.remove('params.problem.w_max_inter')
    # problem_columns.remove('params.problem.w_min_inter')
    problem_columns = [c for c in problem_columns if c not in ['params.problem.normalize', 'params.problem.graph_type_inter']] #, 'params.problem.w_max_intra', 'params.problem.w_min_intra']]
    solver_columns = get_solver_columns_names(df)
    solver_columns = [c for c in solver_columns if c not in ['params.solver.nonzero_threshold', 'params.solver.constraints_mode', 'params.solver.plot', 'params.solver.tabu_edges', 'params.solver.target_mip_gap', 'params.solver.time_limit', 'params.solver.weights_bound', 'params.solver.robust', 'params.solver.callback_mode',]]
    columns_p = problem_columns + solver_columns
    columns_to_plot_default = ['params.problem.graph_type_intra', 'params.problem.intra_edge_ratio', 'params.problem.p','params.problem.noise_scale' , 'params.problem.noise_scale_variance']
    columns_to_plot_in_data = set(problem_columns)
    columns_p_plot = [c for c in columns_to_plot_default if c in columns_to_plot_in_data]

    df = aggregate_data(df, columns_p, columns_m)
    df = filter_best(df, problem_columns, solver_columns, 'metrics.best_shd_mean') #'metrics.t0.5_shd_mean') #

    print_solver_params(df, solver_columns)

    def method_names(x):
        if x['params.problem.noise_scale_variance'] != 'None':
            noise_str = '-var'
        else:
            noise_str = ''
        s = f'{str(x["params.problem.graph_type_intra"]).upper()}-{x['params.problem.intra_edge_ratio']}-{x['params.problem.p']}' + noise_str
        return s

    if graph_type == 'solution_metrics':
        for metric_to_plot, label  in [('metrics.runtime', 'Running Time [s]')]:#,('metrics.best_shd', 'Structural Hamming distance - SHD'),('metrics.t0.5_shd', 'Structural Hamming distance - SHD'), ('metrics.t0.15_shd', 'Structural Hamming distance - SHD'), ('metrics.best_f1score', 'F1 score'), ('metrics.t0.15_f1score', 'F1 score') ]: # ,
            plot_min = 'f1score' in metric_to_plot
            plot_max = 'shd' in metric_to_plot
            make_grid_problem_samples_plot(df, metric_to_plot, label, output_dir, f'icml2025', columns_p_plot, plot_max, plot_min, method_names)
    elif graph_type == 'exdbn_metrics':
        make_grid_problem_samples_plot(df, 'metrics.lazy_added', 'Number of lazy constraints', output_dir, f'icml2025', columns_p_plot, False, False, method_names)


def generate_exdbn_constraints_addition_comparison():
    pl.Config.set_tbl_width_chars(300)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_rows(-1)
    output_dir = '/Users/pavel/0_code/0_causal/data_exdag_exdbn/EXDBN_TMLR_graphs'
    df_old = load_polars_dataframe(join(output_dir, 'exdbn_icml2025'))
    df_old = df_old.filter(pl.col('params.solver.name') == 'milp')
    df_old = df_old.filter((pl.col('params.problem.name') == 'dynamic') & (pl.col('params.problem.generator') == 'notears') & (pl.col('params.problem.graph_type_intra') == 'sf') & (pl.col('params.problem.p') == '1') & (pl.col('params.problem.intra_edge_ratio') == '3'))
    data = [(['EXDBN2'], 'mlflow_planserver', '')]
    #df_additional, _ = experiments_utils.load_data(data)
    #experiments_utils.save_polars_dataframe(df_additional, join(output_dir, 'exdbn_tmlr'))
    df_additional = load_polars_dataframe(join(output_dir, 'exdbn_tmlr'))
    #common_cols = list(set(df.columns) & set(df_additional.columns))
    common_cols = df_additional.columns



    #df = df.select(common_cols)
    df_additional = df_additional.select(common_cols)
    problem_columns = [c for c in common_cols if c.startswith('params.problem')]
    solver_columns = [c for c in common_cols if c.startswith('params.solver')]
    #solver_columns.remove('params.solver.tabu_edges')
    #solver_columns.remove('params.solver.nonzero_threshold')

    df = df_additional


    #df = pl.concat([df, df_additional])
    df = df.with_columns([
        pl.col(c).cast(pl.Int64) for c in ['params.problem.number_of_variables', 'params.problem.number_of_samples']
    ])
    #problem_columns.remove('params.problem.w_max_inter')
    #problem_columns.remove('params.problem.w_min_inter')
    #problem_columns = [c for c in problem_columns if c not in ['params.problem.normalize', 'params.problem.graph_type_inter', 'params.problem.w_max_intra', 'params.problem.w_min_intra']]
    metrics = ['metrics.lazy_added', 'metrics.best_shd','metrics.mipgap']
    columns = problem_columns + solver_columns + metrics
    df = df.select(columns)

    solver_columns.remove('params.solver.callback_mode')
    group_by_columns = problem_columns + solver_columns


    df_p = df.pivot(index=group_by_columns, on='params.solver.callback_mode', values=metrics) #, aggregate_function="len")

    df_p = df_p.with_columns([
        ((pl.col('metrics.best_shd_shortest_cycle') - pl.col('metrics.best_shd_all_cycles')) / pl.col('metrics.best_shd_shortest_cycle')).alias('metrics.all_cycles_to_shortest_cycle'),
        ((pl.col('metrics.best_shd_shortest_cycle') - pl.col('metrics.best_shd_first_cycle')) / pl.col('metrics.best_shd_shortest_cycle')).alias('metrics.first_cycle_to_shortest_cycle'),
        ((pl.col('metrics.best_shd_all_cycles') - pl.col('metrics.best_shd_first_cycle')) / pl.col('metrics.best_shd_all_cycles')).alias('metrics.first_cycle_to_all_cycles'),
    ])


    mean_all_cycles_to_shortes_improvement = df_p.filter(pl.col('metrics.all_cycles_to_shortest_cycle').is_not_nan() & (pl.col("metrics.all_cycles_to_shortest_cycle") != float("inf"))  # remove +inf
    & (pl.col("metrics.all_cycles_to_shortest_cycle") != -float("inf")))['metrics.all_cycles_to_shortest_cycle'].mean()
    print(f'Mean improvement of all cycles to shortest cycle: {mean_all_cycles_to_shortes_improvement}')

    mean_first_cycle_to_shortest_improvement = df_p.filter(pl.col('metrics.first_cycle_to_shortest_cycle').is_not_nan() & (pl.col("metrics.first_cycle_to_shortest_cycle") != float("inf"))  # remove +inf
    & (pl.col("metrics.first_cycle_to_shortest_cycle") != -float("inf")))['metrics.first_cycle_to_shortest_cycle'].mean()
    print(f'Mean improvement of first cycle to shortest cycle: {mean_first_cycle_to_shortest_improvement}')

    mean_first_cycle_to_all_cycles_improvement = df_p.filter(pl.col('metrics.first_cycle_to_all_cycles').is_not_nan() & (pl.col("metrics.first_cycle_to_all_cycles") != float("inf"))  # remove +inf
    & (pl.col("metrics.first_cycle_to_all_cycles") != -float("inf")))['metrics.first_cycle_to_all_cycles'].mean()
    print(f'Mean improvement of first_cycle_to_all_cycles: {mean_first_cycle_to_all_cycles_improvement}')

    print(f'mean mipgap shortest_cycle {df_p['metrics.mipgap_shortest_cycle'].mean()}')
    print(f'mean mipgap all_cycles {df_p['metrics.mipgap_all_cycles'].mean()}')
    print(f'mean mipgap first_cycle {df_p['metrics.mipgap_first_cycle'].mean()}')


    df_agg_mean = df_p.filter(pl.col('metrics.all_cycles_to_shortest_cycle').is_not_nan() & (pl.col("metrics.all_cycles_to_shortest_cycle") != float("inf"))  # remove +inf
    & (pl.col("metrics.all_cycles_to_shortest_cycle") != -float("inf"))).group_by(['params.problem.number_of_variables', 'params.problem.number_of_samples']).agg(
        pl.col('metrics.all_cycles_to_shortest_cycle').mean().alias('metrics.all_cycles_to_shortest_cycle_mean'),
        pl.col('metrics.best_shd_shortest_cycle').mean().alias('metrics.best_shd_shortest_cycle_mean'),
        pl.col('metrics.best_shd_all_cycles').mean().alias('metrics.best_shd_all_cycles_mean'),

    )

    print(df_agg_mean)




    #print(df_p)

    # dupl = df_p.filter(pl.col('metrics.lazy_added_first_cycle') > 1)
    # print(dupl)

    # max_len =agg_df['metrics.best_shd_len'].max()
    # print(max_len)


def generate_scalability_graph(output_dir, reload_data):
    data = [(['EXDBNSCALABILITY'], 'mlflow_planserver', '')]
    if reload_data:
        df, _ = experiments_utils.load_data(data, csv_file_path='/Users/pavel/0_code/0_causal/data_exdag_exdbn')
        df = df.drop_nulls(subset=['metrics.best_shd'])
        if "metrics.infeasible" in df.columns:
            df = df.filter((pl.col("metrics.infeasible") != 1) | pl.col("metrics.infeasible").is_null())
        experiments_utils.save_polars_dataframe(df, join(output_dir, 'exdbn_tmlr_scalability'))
    else:
        df = experiments_utils.load_polars_dataframe(join(output_dir, 'exdbn_tmlr_scalability'))
    df = df.with_columns([
        pl.col("params.problem.number_of_variables").cast(pl.Int64),
        pl.col("params.problem.number_of_samples").cast(pl.Int64)
    ])
    df = df.filter(pl.col('params.problem.number_of_variables') <= 80)
    df = df.select(['params.solver.name','params.problem.number_of_variables', 'params.problem.number_of_samples','params.problem.w_max_inter','metrics.best_f1score',  'metrics.best_shd', 'metrics.runtime', 'metrics.t0.5_shd', 'metrics.t0.15_shd', 'metrics.t0.15_f1score', 'metrics.t0.5_f1score', 'metrics.lazy_added'])
    #df = df.filter(pl.col('params.problem.w_max_inter') == '0.3')
    #df = df.filter(pl.col('params.solver.name') == 'milp')
    df = df.group_by(['params.problem.number_of_variables', 'params.problem.number_of_samples', 'params.solver.name']).median()
    print(df)
    plt.scatter(df['params.problem.number_of_variables'], df['metrics.t0.15_shd'])
    plt.show()
    # px.scatter(df, x="params.problem.number_of_variables", y="metrics.t0.15_shd", color="params.problem.w_max_inter",
    #            title="Colored scatter plot").show()
    # alt.renderers.enable('mimetype')

    # base = (
    #     alt.Chart(data_long)
    #     .encode(
    #         x=alt.X("Date:T", title="Datum"),
    #         y=alt.Y("revenue:Q", title="Tržby [tisíc EUR]", scale=alt.Scale(domainMin=0)),  # force y-axis to start at 0
    #         color="series:N"
    #     )
    # )
    #
    # chart = base.mark_line(strokeWidth=1) + base.mark_point(size=60, filled=True)
    df = replace_dots_in_columns(df)
    print(df)
    chart = alt.Chart(df).encode(
        x=alt.X("params_problem_number_of_variables:N"),
        y=alt.Y("metrics_t0_15_shd:Q"),
        color="params_solver_name:N"
    ).mark_line(strokeWidth=1)
    chart.show()

    df = df.sort("params_problem_number_of_variables")
    solvers = df.select("params_solver_name").unique().to_series().to_list()

    # Create a consistent color map
    colors = plt.get_cmap("tab10")#, len(solvers))
    solver_colors = {solver: colors(i) for i, solver in enumerate(solvers)}
    fig, ax = plt.subplots(figsize=(4, 3))

    for solver, group in df.group_by("params_solver_name", maintain_order=True):
        ax.plot(
            group["params_problem_number_of_variables"].to_numpy(),
            group["metrics_t0_15_shd"].to_numpy(),
            label=solver[0],
            color=solver_colors[solver[0]],
            linewidth=1
        )

    ax.set_xlabel("Number of variables")
    ax.set_ylabel("SHD")
    ax.legend(title="Algorithm")
    fig.savefig(join(output_dir,"scalability_sf3_shd_300.png"), dpi=300, bbox_inches="tight")
    plt.show()

    fig, ax = plt.subplots(figsize=(4, 3))

    for solver, group in df.group_by("params_solver_name", maintain_order=True):
        ax.plot(
            group["params_problem_number_of_variables"].to_numpy(),
            group["metrics_t0_15_f1score"].to_numpy(),
            label=solver[0],
            color=solver_colors[solver[0]],
            linewidth=1
        )

    ax.set_xlabel("Number of variables")
    ax.set_ylabel("F1 score")
    ax.legend(title="Algorithm")
    fig.savefig(join(output_dir, "scalability_sf3_f1_300.png"), dpi=300, bbox_inches="tight")
    plt.show()

def create_convergence_graphs(output_dir, reload_data):
    data = [(['EXDBN_CONVERGENCE'], 'mlflow_planserver', '')]
    if reload_data:
        df, _ = experiments_utils.load_data(data, csv_file_path='/Users/pavel/0_code/0_causal/data_exdag_exdbn')
        df = df.drop_nulls(subset=['metrics.best_shd'])
        if "metrics.infeasible" in df.columns:
            df = df.filter((pl.col("metrics.infeasible") != 1) | pl.col("metrics.infeasible").is_null())
        experiments_utils.save_polars_dataframe(df, join(output_dir, 'exdbn_tmlr_convergence'))
    else:
        df = experiments_utils.load_polars_dataframe(join(output_dir, 'exdbn_tmlr_convergence'))
    df = df.with_columns([
        pl.col("params.solver.time_limit").cast(pl.Int64),
        (pl.col("params.problem.graph_type_intra").str.to_uppercase() + '-' + pl.col("params.problem.intra_edge_ratio") + '-' + pl.col("params.problem.p")).alias('problem')
    ])
    df = df.select(["params.solver.time_limit", "problem", "metrics.t0.15_shd"])
    print(df)
    problems = df["problem"].unique().to_list()
    n_problems = len(problems)

    n_cols = 2 #math.ceil(math.sqrt(n_problems))
    n_rows = math.ceil(n_problems / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)

    for i, problem in enumerate(problems):
        r, c = divmod(i, n_cols)
        ax = axes[r, c]

        sub_df = df.filter(pl.col("problem") == problem).sort("params.solver.time_limit")

        ax.plot(sub_df["params.solver.time_limit"], sub_df["metrics.t0.15_shd"], label=str(problem))

        ax.set_title(f"{problem}")
        ax.set_xlabel("Time limit")
        ax.set_ylabel("SHD")
        #ax.grid(True, linestyle="--", alpha=0.5)
        #ax.legend()

    # Hide empty subplots if any
    for j in range(i + 1, n_rows * n_cols):
        r, c = divmod(j, n_cols)
        axes[r, c].axis("off")

    plt.tight_layout()
    plt.savefig(join(output_dir, "convergence.png"), dpi=300, bbox_inches="tight")
    plt.show()


def create_variance_graphs(output_dir, reload_data):
    data = [(['EXDBN_VARIANCE'], 'mlflow_planserver', '')]
    if reload_data:
        df, _ = experiments_utils.load_data(data, csv_file_path='/Users/pavel/0_code/0_causal/data_exdag_exdbn')
        df = df.drop_nulls(subset=['metrics.best_shd'])
        if "metrics.infeasible" in df.columns:
            df = df.filter((pl.col("metrics.infeasible") != 1) | pl.col("metrics.infeasible").is_null())
        experiments_utils.save_polars_dataframe(df, join(output_dir, 'exdbn_tmlr_variance'))
    else:
        df = experiments_utils.load_polars_dataframe(join(output_dir, 'exdbn_tmlr_variance'))
    df = df.with_columns([
        pl.col("params.solver.time_limit").cast(pl.Int64),
        pl.col("params.problem.seed").cast(pl.Int64),
        (pl.col("params.problem.graph_type_intra").str.to_uppercase() + '-' + pl.col("params.problem.intra_edge_ratio") + '-' + pl.col("params.problem.p")).alias('problem')
    ])
    df = df.select(["params.solver.time_limit", "problem", "metrics.t0.15_shd", "params.problem.seed"])
    df = df.filter(pl.col("params.problem.seed") < 10)
    print(df)
    problems = df["problem"].unique().to_list()
    n_problems = len(problems)

    n_cols = 2 #math.ceil(math.sqrt(n_problems))
    n_rows = math.ceil(n_problems / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)

    for i, problem in enumerate(problems):
        r, c = divmod(i, n_cols)
        ax = axes[r, c]

        sub_df = df.filter(pl.col("problem") == problem).sort("params.solver.time_limit")

        agg = (
            sub_df
            .group_by("params.solver.time_limit")
            .agg([
                pl.col("metrics.t0.15_shd").mean().alias("shd_mean"),
                pl.col("metrics.t0.15_shd").std().alias("shd_std"),
            ])
            .sort("params.solver.time_limit")
        )

        x = agg["params.solver.time_limit"].to_list()
        y = agg["shd_mean"].to_list()
        y_std = agg["shd_std"].to_list()

        # choose a color per subplot (problem) for consistency
        # ax_handles, ax_labels = ax.get_legend_handles_labels()
        # for (g_a, method_data) in group_data.groupby('Method', dropna=False):
        #     color = ax_handles[ax_labels.index(g_a)].get_color()
        # line_color = next(ax._get_lines.prop_cycler)["color"] if hasattr(ax, "_get_lines") else "C0"
        ax.plot(x, y, linewidth=1.5, label=str(problem))
        ax.fill_between(
            x,
            [a - (b if b is not None else 0.0) for a, b in zip(y, y_std)],
            [a + (b if b is not None else 0.0) for a, b in zip(y, y_std)],
            #color=line_color,
            alpha=0.25,
            linewidth=0
        )


        # for seed, seed_group in sub_df.group_by("params.problem.seed", maintain_order=True):
        #     ax.plot(
        #         seed_group["params.solver.time_limit"],
        #         seed_group["metrics.t0.15_shd"],
        #         label=f"{problem} | seed={seed[0]}",
        #         marker="o",
        #         markersize=3,
        #         linewidth=1
        #     )
        #ax.plot(sub_df["params.solver.time_limit"], sub_df["metrics.t0.15_shd"], label=str(problem))

        ax.set_title(f"{problem}")
        ax.set_xlabel("Time limit")
        ax.set_ylabel("SHD")
        #ax.grid(True, linestyle="--", alpha=0.5)
        #ax.legend()

    # Hide empty subplots if any
    for j in range(i + 1, n_rows * n_cols):
        r, c = divmod(j, n_cols)
        axes[r, c].axis("off")

    plt.tight_layout()
    plt.savefig(join(output_dir, "convergence_variance.png"), dpi=300, bbox_inches="tight")
    plt.show()



if __name__ == "__main__":
    output_dir = '/Users/pavel/0_code/0_causal/data_exdag_exdbn/EXDBN_TMLR_graphs'  # '/Users/pavel/0_code/0_causal/data_exdag_exdbn/EXDBN_ICML2025_graphs'
    #create_variance_graphs(output_dir, True)
    #create_convergence_graphs(output_dir, False)
    #generate_scalability_graph(output_dir, False)
    #exit(0)
    # generate_exdbn_constraints_addition_comparison()
    # exit(0)
    generate_exdbn_graphs(output_dir,'solution_metrics', False)
    #generate_exdbn_graphs(output_dir, 'exdbn_metrics')



