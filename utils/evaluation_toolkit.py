import torch
from torchmetrics import AUROC, AveragePrecision
import numpy as np
from sklearn import metrics as sklearn_metrics
from .constants import *

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