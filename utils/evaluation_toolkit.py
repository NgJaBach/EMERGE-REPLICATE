import torch
from torchmetrics import AUROC, AveragePrecision
import numpy as np
from sklearn import metrics as sklearn_metrics
from .constants import *
from pathlib import Path
import math
import matplotlib.pyplot as plt

def minpse(preds, labels):
    precisions, recalls, thresholds = sklearn_metrics.precision_recall_curve(labels, preds)
    minpse_score = np.max([min(x, y) for (x, y) in zip(precisions, recalls)])
    return minpse_score

def get_binary_metrics(preds, labels):
    auroc = AUROC(task="binary")
    auprc = AveragePrecision(task="binary")

    # convert labels type to int
    labels = labels.type(torch.int)
    auroc(preds, labels)
    auprc(preds, labels)
    minpse_score = minpse(preds, labels) 

    return {
        "auroc": auroc.compute().item(),
        "auprc": auprc.compute().item(),
        "minpse": minpse_score,
    }

def get_all_metrics(y_outcome_pred, y_readmission_pred, y_outcome_true, y_readmission_true):
    outcome_metrics = get_binary_metrics(y_outcome_pred, y_outcome_true)
    readmission_metrics = get_binary_metrics(y_readmission_pred, y_readmission_true)
    # Merging with prefixes
    merged_dict = {f"outcome_{k}": v for k, v in outcome_metrics.items()}
    merged_dict.update({f"readmission_{k}": v for k, v in readmission_metrics.items()})
    return merged_dict

def check_metric_is_better(cur_best, score, main_metric='outcome_auroc'):
    if cur_best == {}:
        return True
    if score > cur_best[main_metric]:
        return True
    return False

def bootstrap(preds_outcome, preds_readmission, labels_outcome, labels_readmission, K=10, seed=RANDOM_STATE):
    """Bootstrap resampling for binary classification metrics. Resample K times"""
    
    length = len(preds_outcome)
    np.random.seed(seed)
    
    # Initialize a list to store bootstrap samples
    bootstrapped_samples = []

    # Create K bootstrap samples
    for _ in range(K):
        # Sample with replacement from the indices
        sample_indices = np.random.choice(length, length, replace=True)

        # Use the sampled indices to get the bootstrap sample of preds and labels
        sample_preds_outcome = preds_outcome[sample_indices]
        sample_preds_readmission = preds_readmission[sample_indices]
        sample_labels_outcome = labels_outcome[sample_indices]
        sample_labels_readmission = labels_readmission[sample_indices]
        
        # Store the bootstrap samples
        bootstrapped_samples.append((sample_preds_outcome, sample_preds_readmission, sample_labels_outcome, sample_labels_readmission))

    return bootstrapped_samples

def export_metrics(bootstrapped_samples):
    metrics = { "outcome_auroc": [], "outcome_auprc": [], "outcome_minpse": [], "readmission_auroc": [], "readmission_auprc": [], "readmission_minpse": []}
    
    for sample in bootstrapped_samples:
        sample_preds_outcome, sample_preds_readmission, sample_labels_outcome, sample_labels_readmission = sample[0], sample[1], sample[2], sample[3]
        res = get_all_metrics(sample_preds_outcome, sample_preds_readmission, sample_labels_outcome, sample_labels_readmission)

        for k, v in res.items():
            metrics[k].append(v)

    # convert to numpy array
    for k, v in metrics.items():
        metrics[k] = np.array(v)
    
    # calculate mean and std
    for k, v in metrics.items():
        metrics[k] = {"mean": np.mean(v), "std": np.std(v)}
    return metrics

def run_bootstrap(preds_outcome, preds_readmission, labels_outcome, labels_readmission, seed=42):
    bootstrap_samples = bootstrap(preds_outcome, preds_readmission, labels_outcome, labels_readmission, seed=seed)
    metrics = export_metrics(bootstrap_samples)
    return metrics

def save_per_label_accuracy(preds, labels, save_dir, prefix, threshold=0.5):
    # tensors
    p = torch.as_tensor(preds).float().squeeze()
    y = torch.as_tensor(labels).int().squeeze()

    # positive-class probability
    if p.ndim == 2 and p.size(1) == 2:
        ppos = torch.softmax(p, dim=1)[:, 1] if (p.min() < 0 or p.max() > 1) else p[:, 1]
    else:
        ppos = torch.sigmoid(p) if (p.min() < 0 or p.max() > 1) else p

    yhat = (ppos >= threshold).int()

    tn = int(((yhat == 0) & (y == 0)).sum())
    tp = int(((yhat == 1) & (y == 1)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum())

    n0, n1 = tn + fp, tp + fn
    tnr = tn / n0 if n0 else float("nan")  # label 0 % correct
    tpr = tp / n1 if n1 else float("nan")  # label 1 % correct
    bal_acc = (tnr + tpr) / 2 if (n0 and n1) else float("nan")
    prevalence = (n1 / (n0 + n1)) if (n0 + n1) else float("nan")

    # ensure output dir
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # plot
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar([0, 1], [tnr * 100, tpr * 100])
    ax.set_xticks([0, 1], ['label 0', 'label 1'])
    ax.set_ylim(0, 100)
    ax.set_ylabel('% correct')
    ax.set_title(f'Per-label accuracy (thr={threshold:.2f})\nBalanced acc: {bal_acc*100 if not math.isnan(bal_acc) else float("nan"):.1f}%')
    for i, (pct, corr, tot) in enumerate([(tnr, tn, n0), (tpr, tp, n1)]):
        if tot:
            ax.text(i, pct * 100 + 1, f'{pct*100:.1f}%\n({corr}/{tot})',
                    ha='center', va='bottom', fontsize=8)
    fig.tight_layout()
    plot_path = save_dir / f"{prefix}_per_label_accuracy.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    # text summary
    def fmt(x): 
        return f"{x:.6f}" if isinstance(x, float) and not math.isnan(x) else "nan"

    summary = (
        f"threshold: {threshold}\n"
        f"counts: tn={tn}, fp={fp}, fn={fn}, tp={tp}\n"
        f"n0={n0}, n1={n1}, prevalence_pos={fmt(prevalence)}\n"
        f"per-label: label0_TNR={fmt(tnr)}, label1_TPR={fmt(tpr)}\n"
        f"balanced_accuracy={fmt(bal_acc)}\n"
        f"plot_file={plot_path.name}\n"
    )
    txt_path = save_dir / f"{prefix}_per_label_metrics.txt"
    with open(txt_path, "w") as f:
        f.write(summary)

    # JSON (use None instead of NaN for validity)
    def safe(x):
        return None if (isinstance(x, float) and (math.isnan(x) or math.isinf(x))) else x

    data = {
        "threshold": float(threshold),
        "counts": {"tn": tn, "fp": fp, "fn": fn, "tp": tp, "n0": n0, "n1": n1},
        "pct_correct": {"label0_TNR": safe(tnr), "label1_TPR": safe(tpr)},
        "balanced_accuracy": safe(bal_acc),
        "prevalence_pos": safe(prevalence),
        "plot_file": str(plot_path),
        "text_file": str(txt_path),
    }

    return data
