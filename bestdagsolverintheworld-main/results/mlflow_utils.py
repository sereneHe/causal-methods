
def convert_to_df(runs):

    import numpy as np
    import pandas as pd

    info = {
        "run_id": [],
        "experiment_id": [],
        "status": [],
        "artifact_uri": [],
        "start_time": [],
        "end_time": [],
    }
    params, metrics, tags = ({}, {}, {})
    PARAM_NULL, METRIC_NULL, TAG_NULL = (None, np.nan, None)
    for i, run in enumerate(runs):
        info["run_id"].append(run.info.run_id)
        info["experiment_id"].append(run.info.experiment_id)
        info["status"].append(run.info.status)
        info["artifact_uri"].append(run.info.artifact_uri)
        info["start_time"].append(pd.to_datetime(run.info.start_time, unit="ms", utc=True))
        info["end_time"].append(pd.to_datetime(run.info.end_time, unit="ms", utc=True))

        # Params
        param_keys = set(params.keys())
        for key in param_keys:
            if key in run.data.params:
                params[key].append(run.data.params[key])
            else:
                params[key].append(PARAM_NULL)
        new_params = set(run.data.params.keys()) - param_keys
        for p in new_params:
            params[p] = [PARAM_NULL] * i  # Fill in null values for all previous runs
            params[p].append(run.data.params[p])

        # Metrics
        metric_keys = set(metrics.keys())
        for key in metric_keys:
            if key in run.data.metrics:
                metrics[key].append(run.data.metrics[key])
            else:
                metrics[key].append(METRIC_NULL)
        new_metrics = set(run.data.metrics.keys()) - metric_keys
        for m in new_metrics:
            metrics[m] = [METRIC_NULL] * i
            metrics[m].append(run.data.metrics[m])

        # Tags
        tag_keys = set(tags.keys())
        for key in tag_keys:
            if key in run.data.tags:
                tags[key].append(run.data.tags[key])
            else:
                tags[key].append(TAG_NULL)
        new_tags = set(run.data.tags.keys()) - tag_keys
        for t in new_tags:
            tags[t] = [TAG_NULL] * i
            tags[t].append(run.data.tags[t])

    data = {}
    data.update(info)
    for key, value in metrics.items():
        data["metrics." + key] = value
    for key, value in params.items():
        data["params." + key] = value
    for key, value in tags.items():
        data["tags." + key] = value
    return pd.DataFrame(data)