import sys
from typing import List

def fetch_neptune_data(tags: List[str]):
    import neptune
    project = neptune.init_project(
        mode="read-only",
    )
    runs_table_df = project.fetch_runs_table(tag=tags).to_pandas()
    return runs_table_df

def export_neptune_to_csv(experiment_name: str, filename: str):
    runs_table_df = fetch_neptune_data([experiment_name])
    runs_table_df.to_csv(filename, index=False)

if __name__ == "__main__":
    export_neptune_to_csv(sys.argv[1], sys.argv[2])

def rename_neptune_columns(df):
    if 'sys/id' in df.columns.values:
        columns_to_rename = {c: c.replace('/','.') for c in df.columns.values if ('params' in c) or ('metrics' in c)}
        columns_to_rename['sys/id'] = 'run_id'
        columns_to_rename['sys/tags'] = 'experiment_id'
        #columns_to_rename['sys/creation_time'] = 'start_time'

        df.rename(columns=columns_to_rename, inplace=True)
        #df['experiment_id'] = df['run_id'].apply(lambda x: x.split('-')[0])
        columns_to_drop = [c for c in df.columns.values if c.startswith('source_code') or c.startswith('monitoring') or c.startswith('sys')]
        df.drop(columns=columns_to_drop, inplace=True)


    return df