import os
import glob
import torch
import numpy as np
import pandas as pd
from math import sqrt

LOG_ROOT = "logs"

def mean_ci(x, alpha=0.95):
    x = np.array(x, dtype=float)
    if len(x) == 0:
        return np.nan, np.nan, np.nan, np.nan
    mean = x.mean()
    # sample std (ddof=1) if len>1, else 0
    std = x.std(ddof=1) if len(x) > 1 else 0.0
    se = std / sqrt(len(x)) if len(x) > 0 else np.nan
    z = 1.96  # 95% CI for normal approx
    return mean, std, se, z * se

def infer_experiment_name_from_dir(dir_name: str) -> str:
    """
    Map directory names like:
      - 'baseline_1'      -> 'baseline'
      - 'drop_high_2'     -> 'drop_high'
      - 'drop_low_3'      -> 'drop_low'
      - 'frontloaded_1'   -> 'frontloaded'
      - 'constant_2'      -> 'constant'
    """
    parts = dir_name.split("_")

    if parts[0] == "drop" and len(parts) >= 2:
        # drop_high_1 -> drop_high
        return parts[0] + "_" + parts[1]
    else:
        # baseline_1 -> baseline, frontloaded_2 -> frontloaded, constant_3 -> constant
        return parts[0]

experiments = {}

for log_path in glob.glob(os.path.join(LOG_ROOT, "*", "log.pt")):
    dir_name = os.path.basename(os.path.dirname(log_path))
    exp_name = infer_experiment_name_from_dir(dir_name)

    log = torch.load(log_path, map_location="cpu", weights_only=False)

    accs = np.array(log["results"]["accs"], dtype=float)

    gpu_times = np.array(log["results"]["gpu_times"], dtype=float)
    wall_times = np.array(log["results"]["wall_times"], dtype=float)

    meta = log["metadata"]

    if exp_name not in experiments:
        experiments[exp_name] = {
            "accs": [],
            "gpu_times": [],
            "wall_times": [],
            "epochs": meta["num_epochs"],
            "batch_size": meta["batch_size"],
            "git_commit": meta["git_commit"][:8],
        }

    experiments[exp_name]["accs"].extend(accs.tolist())
    experiments[exp_name]["gpu_times"].extend(gpu_times.tolist())
    experiments[exp_name]["wall_times"].extend(wall_times.tolist())

rows = []

for exp_name, data in experiments.items():
    acc_mean, acc_std, acc_se, acc_ci = mean_ci(data["accs"])
    gpu_mean, gpu_std, gpu_se, gpu_ci = mean_ci(data["gpu_times"])
    wall_mean, wall_std, wall_se, wall_ci = mean_ci(data["wall_times"])

    row = {
        "experiment": exp_name,
        "num_runs": len(data["accs"]),

        "mean_acc": acc_mean,
        "std_acc": acc_std,
        "stderr_acc": acc_se,
        "ci95_acc": acc_ci,

        "gpu_time_mean": gpu_mean,
        "gpu_time_std": gpu_std,
        "gpu_time_stderr": gpu_se,
        "gpu_time_ci95": gpu_ci,

        "wall_time_mean": wall_mean,
        "wall_time_std": wall_std,
        "wall_time_stderr": wall_se,
        "wall_time_ci95": wall_ci,

        "epochs": data["epochs"],
        "batch_size": data["batch_size"],
        "git_commit": data["git_commit"],
    }
    rows.append(row)

df = pd.DataFrame(rows)

df = df.sort_values(by=["mean_acc", "wall_time_mean"], ascending=[False, True])

print(df.to_string(index=False))

df.to_csv("summary_table_timing.csv", index=False)
print("saved")
