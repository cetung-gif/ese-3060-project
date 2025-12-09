import os
import glob
import torch
import numpy as np
import pandas as pd
from math import sqrt

LOG_ROOT = "logs"
OUT_CSV = "summary_table.csv"

def mean_ci(x, alpha=0.95):
    """
    Returns mean, std, stderr, and 95% CI half-width
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    mean = x.mean()
    std = x.std(ddof=1)
    se = std / sqrt(n)
    z = 1.96  # 95% CI (normal approx, fine for n>=10)
    return mean, std, se, z * se

rows = []

log_paths = sorted(glob.glob(os.path.join(LOG_ROOT, "*", "log.pt")))
assert len(log_paths) > 0, "no logs found"

for log_path in log_paths:
    log = torch.load(log_path, map_location="cpu", weights_only=False)

    accs = log["results"]["accs"]
    mean, std, se, ci = mean_ci(accs)

    meta = log["metadata"]
    results = log["results"]

    row = {
        # identifiers
        "experiment": log["experiment"],
        "git_commit": meta["git_commit"][:8],

        # sample size
        "num_runs": meta["num_runs"],

        # accuracy stats
        "mean_acc": mean,
        "std_acc": std,
        "stderr": se,
        "ci95": ci,
        "ci95_low": mean - ci,
        "ci95_high": mean + ci,

        # timing stats
        "gpu_time_mean": results.get("gpu_time_mean", np.nan),
        "wall_time_mean": results.get("wall_time_mean", np.nan),

        # hyperparameters (important for reproducibility)
        "batch_size": meta["batch_size"],
        "epochs": meta["num_epochs"],
    }

    rows.append(row)

df = pd.DataFrame(rows)

df["acc_rank"] = df["mean_acc"].rank(ascending=False)
df["speed_per_epoch"] = df["wall_time_mean"] / df["epochs"]

df = df.sort_values("mean_acc", ascending=False)

pd.set_option("display.precision", 4)
print("\n=== EXPERIMENT SUMMARY ===\n")
print(df.to_string(index=False))

df.to_csv(OUT_CSV, index=False)
print(f"\nSaved summary table to: {OUT_CSV}")
