"""
Unsupervised clustering metrics: Silhouette, Calinski-Harabasz, Davies-Bouldin,
Adjusted Rand Index, Normalized Mutual Information, Purity.
"""
from __future__ import annotations
from typing import Dict, Optional
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def purity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # assumes labels are integers starting at 0
    if y_true is None:
        return float('nan')
    labels = np.unique(y_pred)
    N = len(y_true)
    total = 0
    for k in labels:
        idx = y_pred == k
        if idx.sum() == 0:
            continue
        true_labels, counts = np.unique(y_true[idx], return_counts=True)
        total += counts.max()
    return total / N


def compute_all_metrics(Z: np.ndarray, y_pred: np.ndarray, y_true: Optional[np.ndarray] = None) -> Dict[str, float]:
    out: Dict[str, float] = {}
    # guard: DBSCAN may yield -1 noise labels; filter for indices with labels >=0 for silhouette/CH/DB
    valid = y_pred >= 0
    if valid.sum() > 1 and len(np.unique(y_pred[valid])) > 1:
        try:
            out['silhouette'] = float(silhouette_score(Z[valid], y_pred[valid]))
        except Exception:
            out['silhouette'] = float('nan')
        try:
            out['calinski_harabasz'] = float(calinski_harabasz_score(Z[valid], y_pred[valid]))
        except Exception:
            out['calinski_harabasz'] = float('nan')
        try:
            out['davies_bouldin'] = float(davies_bouldin_score(Z[valid], y_pred[valid]))
        except Exception:
            out['davies_bouldin'] = float('nan')
    else:
        out['silhouette'] = float('nan')
        out['calinski_harabasz'] = float('nan')
        out['davies_bouldin'] = float('nan')

    if y_true is not None and len(y_true) == len(y_pred):
        try:
            out['ari'] = float(adjusted_rand_score(y_true, y_pred))
        except Exception:
            out['ari'] = float('nan')
        try:
            out['nmi'] = float(normalized_mutual_info_score(y_true, y_pred))
        except Exception:
            out['nmi'] = float('nan')
        try:
            out['purity'] = float(purity_score(y_true, y_pred))
        except Exception:
            out['purity'] = float('nan')
    else:
        out['ari'] = float('nan')
        out['nmi'] = float('nan')
        out['purity'] = float('nan')
    return out
