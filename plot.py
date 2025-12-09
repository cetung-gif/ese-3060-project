import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt

LOG_ROOT = "logs"

def collect_histories(log):
    return log["histories"]

def aggregate_curve(histories, key):
    max_len = max(len(h[key]) for h in histories)
    vals = [[] for _ in range(max_len)]

    for h in histories:
        for i, v in enumerate(h[key]):
            vals[i].append(v)

    mean = np.array([np.mean(v) for v in vals])
    std = np.array([np.std(v) for v in vals])
    return mean, std

plt.figure(figsize=(10, 5))

for log_path in glob.glob(os.path.join(LOG_ROOT, "*", "log.pt")):
    log = torch.load(log_path, map_location="cpu", weights_only=False)
    histories = collect_histories(log)

    mean_acc, std_acc = aggregate_curve(histories, "val_acc")
    epochs = np.arange(len(mean_acc))

    plt.plot(epochs, mean_acc, label=log["experiment"])
    plt.fill_between(
        epochs,
        mean_acc - std_acc,
        mean_acc + std_acc,
        alpha=0.2
    )

plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("CIFAR-10 Convergence (mean ± std)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("val_acc_convergence.png", dpi=200)
print("complete")
