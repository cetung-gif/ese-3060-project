import os
import glob
import torch
import numpy as np
import pandas as pd
from math import sqrt

LOG_ROOT = "logs"

def mean_stats(x):
    x = np.asarray(x, dtype=float)
    mean = x.mean()
    std = x.std(ddof=1)
    se = std / sqrt(len(x))
    ci95 = 1.96 * se
    return mean, std, se, ci95

rows = []

for log_path in glob.glob(os.path.join(LOG_ROOT, "*", "log.pt")):
    log = torch.load(log_path, map_location="cpu", weights_only=False)

    dirname = os.path.basename(os.path.dirname(log_path))
    exp = dirname.split("_")[0]

    accs = log["results"]["accs"]
    gpu_times = log["results"]["gpu_times"]
    wall_times = log["results"]["wall_times"]

    acc_mean, acc_std, acc_se, acc_ci = mean_stats(accs)
    gpu_mean, gpu_std, gpu_se, gpu_ci = mean_stats(gpu_times)
    wall_mean, wall_std, wall_se, wall_ci = mean_stats(wall_times)

    rows.append({
        "experiment": exp,
        "num_runs": len(accs),

        "mean_acc": acc_mean,
        "std_acc": acc_std,
        "acc_ci95": acc_ci,

        "gpu_time_mean": gpu_mean,
        "gpu_time_std": gpu_std,
        "gpu_time_ci95": gpu_ci,

        "wall_time_mean": wall_mean,
        "wall_time_std": wall_std,
        "wall_time_ci95": wall_ci,

        "epochs": log["metadata"]["num_epochs"],
        "batch_size": log["metadata"]["batch_size"],
        "git_commit": log["metadata"]["git_commit"][:8],
    })

df = pd.DataFrame(rows)
df = df.sort_values("wall_time_mean")

print(df.to_string(index=False))
df.to_csv("summary_table.csv", index=False)

print("\nSaved summary_table.csv")
