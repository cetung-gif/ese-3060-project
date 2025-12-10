import os
import glob
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG_ROOT = "logs"
OUT_FILE = "convergence_plot.png"

def flatten_histories(histories):
    flat = []
    for h in histories:
        if isinstance(h, dict):
            flat.append(h)
        elif isinstance(h, list):
            flat.extend(h)
        else:
            raise TypeError(f"Unexpected history type: {type(h)}")
    return flat

groups = {}

for log_path in glob.glob(os.path.join(LOG_ROOT, "*", "log.pt")):
    dirname = os.path.basename(os.path.dirname(log_path))

    # experiment name = directory prefix before last underscore
    if "_" in dirname:
        exp_name = "_".join(dirname.split("_")[:-1])
    else:
        exp_name = dirname

    log = torch.load(log_path, map_location="cpu", weights_only=False)

    groups.setdefault(exp_name, []).extend(log["histories"])

plt.figure(figsize=(9, 5))

for exp, histories in sorted(groups.items()):
    histories = flatten_histories(histories)

    max_len = min(len(h["val_acc"]) for h in histories)

    vals = np.array([
        h["val_acc"][:max_len] for h in histories
    ]) 
    mean = vals.mean(axis=0)
    std = vals.std(axis=0, ddof=1)
    ci95 = 1.96 * std / np.sqrt(vals.shape[0])

    epochs = np.arange(max_len)

    plt.plot(epochs, mean, label=exp, linewidth=2)
    plt.fill_between(
        epochs,
        mean - ci95,
        mean + ci95,
        alpha=0.15
    )

plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("CIFAR-10 Convergence (mean ± 95% CI)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_FILE, dpi=200)

print(f"Saved {OUT_FILE}")
