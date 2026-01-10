"""
Clustering utilities and baselines for latent representations and raw features.
"""
from __future__ import annotations
from typing import Dict, Tuple, Optional
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .evaluation import compute_all_metrics


def run_kmeans(Z: np.ndarray, n_clusters: int, random_state: int = 42) -> np.ndarray:
    return KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state).fit_predict(Z)


def run_agglomerative(Z: np.ndarray, n_clusters: int) -> np.ndarray:
    return AgglomerativeClustering(n_clusters=n_clusters, linkage='ward').fit_predict(Z)


def run_dbscan(Z: np.ndarray, eps: float = 0.7, min_samples: int = 5) -> np.ndarray:
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(Z)


def pca_kmeans(X: np.ndarray, n_components: int, n_clusters: int) -> Tuple[np.ndarray, np.ndarray]:
    Xs = StandardScaler().fit_transform(X)
    Z = PCA(n_components=n_components, random_state=42).fit_transform(Xs)
    labels = run_kmeans(Z, n_clusters)
    return Z, labels


def evaluate_clustering(Z: np.ndarray, labels: np.ndarray, y_true: Optional[np.ndarray] = None) -> Dict[str, float]:
    return compute_all_metrics(Z, labels, y_true)


def autoencoder_kmeans(X: np.ndarray, encoder_fn, n_clusters: int):
    Z = encoder_fn(X)
    labels = run_kmeans(Z, n_clusters)
    return Z, labels
